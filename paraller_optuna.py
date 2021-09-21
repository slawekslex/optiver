from fastai.tabular.all import *
from sklearn.model_selection import KFold, GroupKFold
import optuna
from optuna.integration import FastAIPruningCallback
import sqlalchemy

STOCK_COUNT = 112
FEATURE_COUNT = 144

def fill_missing(train_df):
    all_times = train_df.time_id.unique()
    all_stocks = train_df.stock_id.unique()
    filled_df = train_df.copy()
    filled_df=filled_df.set_index(['time_id', 'stock_id'])
    new_index = pd.MultiIndex.from_product([all_times, all_stocks], names = ['time_id', 'stock_id'])
    filled_df = filled_df.reindex(new_index).reset_index()
    filled_df = filled_df.fillna(0)
    return filled_df


    

class MaskTfm(ItemTransform):
    do_tranform=0
    def mask(self, x, indices):
        x[torch.tensor(indices, device=x.device)] = 0
        return x
    
    def __call__(self, b, split_idx=None, **kwargs):
        self.do_transform = (split_idx == 0)
        return super().__call__(b, split_idx=split_idx, **kwargs)
    

    def encodes(self, x):
        if not self.do_transform: return x
        n = len(x[0])
        indices = np.random.choice(np.array(range(n)), n//10, replace=False)
        x = [self.mask(y, indices) for y in x]
        
        return x

class MyDataLoader(TabDataLoader):
    def __init__(self, dataset, bs=16, shuffle=False, after_batch=None, num_workers=0, **kwargs):
        if after_batch is None: after_batch = L(TransformBlock().batch_tfms)+ReadTabBatch(dataset) + [MaskTfm()]
        super().__init__(dataset, bs=bs, shuffle=shuffle, after_batch=after_batch, num_workers=num_workers, **kwargs)

    def shuffle_fn(self, idxs):
        idxs = np.array(idxs).reshape(-1,112)
        np.random.shuffle(idxs)
        return idxs.reshape(-1).tolist()

def get_dls(train_df, bs, trn_idx, val_idx):
    cont_nn,cat_nn = cont_cat_split(train_df, max_card=9000, dep_var='target')
    cat_nn=[x for x in cat_nn if not x in ['row_id', 'time_id']]
    
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(train_df, procs_nn, cat_nn, cont_nn, splits=[list(trn_idx), list(val_idx)], y_names='target')
    dls = to_nn.dataloaders(bs=112*100, shuffle=True, dl_type = MyDataLoader)
    dls.train_ds.split_idx=0
    dls.valid_ds.split_idx=1
    return dls


class TimeEncoding(nn.Module):
    def __init__(self, inp_size, bottleneck, p, multiplier):
        super().__init__()
        self.multiplier  = nn.Parameter(torch.tensor(multiplier)) 
        # self.multiplier = multiplier
        self.initial_layers = LinBnDrop(inp_size, bottleneck, act=nn.ReLU(True), p=p, bn=False)
        
        self.concat_layers = nn.Sequential(
            nn.BatchNorm1d(bottleneck * STOCK_COUNT),
            nn.Linear(bottleneck * STOCK_COUNT, inp_size),
            nn.Tanh()
        )
        
    def forward(self, x):
        y = self.initial_layers(x)
        times = y.shape[0] // STOCK_COUNT
        y = y.view(times, -1)
        y = self.concat_layers(y)
   
        y = y.view(times,1,-1).expand(times,STOCK_COUNT,-1).contiguous().view(times*STOCK_COUNT, -1)
        
        return x + y * self.multiplier

class BN(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.num_features = features
        self.bn = nn.BatchNorm1d(STOCK_COUNT * self.num_features)
    def forward(self, x):
        sh = x.shape
        x = x.view(-1, STOCK_COUNT * self.num_features)
        x = self.bn(x)
        return x.view(*sh)
    
class ParallelModel(nn.Module):
    def __init__(self, inp_size, emb_size, lin_sizes, ps, bottleneck, time_ps, multiplier ):
        super().__init__()
        
        self.stock_emb = nn.Parameter(torch.empty(STOCK_COUNT, emb_size))
        nn.init.normal_(self.stock_emb)
        
        lin_sizes = [inp_size+emb_size] + lin_sizes
        layers = []
        for n_in, n_out, p, time_p in zip(lin_sizes, lin_sizes[1:], ps, time_ps):
            layers.append(nn.Linear(n_in, n_out))
            layers.append(BN(n_out ))
            if p: layers.append(nn.Dropout(p))
            
            layers.append(nn.ReLU(True))
            
            layers.append(TimeEncoding(n_out, bottleneck, time_p, multiplier))
        layers.append(LinBnDrop(lin_sizes[-1], 1, bn=False))
        layers.append(SigmoidRange(0, .1))
        self.layers = nn.Sequential(*layers)
    
    
    def forward(self, x_cat, x_cont):
        times = x_cat.shape[0] // STOCK_COUNT
        s_e = self.stock_emb.repeat(times, 1)
        x = torch.cat([x_cont, s_e], dim=1)
        for l in self.layers.children():
            #print(x.shape, x.mean(), x.std())
            x = l(x)
        return x#self.layers(x)

def rmspe(preds, targs):
    mask = targs != 0
    targs, preds = torch.masked_select(targs, mask), torch.masked_select(preds, mask)
    x = (targs-preds)/targs
    res= (x**2).mean().sqrt()
    if torch.isnan(res): 
        print(targs)
        print(preds)
        raise Exception('fck loss is nan')
    return res

def train(trial, dls, save_as=None):
    inp_size = FEATURE_COUNT
    emb_size = trial.suggest_int('emb_size', 3, 30)
    max_sizes = [2000, 1000, 500]
    lin_sizes = [trial.suggest_int(f'lin_size{i}', 10, ms) for i, ms in enumerate(max_sizes)]
    ps = [0]+[trial.suggest_float(f'p{i}', 0, .8) for i in range(1,3)]
    
    bottleneck = trial.suggest_int('bottleneck', 5, 100)
    time_ps = [trial.suggest_float(f'time_p{i}', 0, .5) for i in range(3)]
    multiplier = trial.suggest_float('multiplier', .01, .5)
    lr = float(trial.suggest_float('lr', 1e-3, 1e-2))
    
    model = ParallelModel(inp_size, emb_size, lin_sizes, ps, bottleneck, time_ps, multiplier)
    #bx1, bx2, by = dls.one_batch()
    
    learn = Learner(dls,model = model, loss_func=rmspe, metrics=AccumMetric(rmspe), opt_func=ranger,
        cbs = FastAIPruningCallback(trial, 'rmspe')).to_fp16()
    # with learn.no_bar():
    #     with learn.no_logging():    
    learn.fit_flat_cos(50, lr)
    if save_as:
        learn.save(save_as)
    last5 = L(learn.recorder.values).itemgot(2)[-5:]
    return np.mean(last5)

def train_cross_valid(trial, dlss, save_as=None):
    res = 0
    for idx, dls in enumerate(dlss):
        v = train(trial, dls, save_as + str(idx) if save_as else None)
        print(f'fold {idx}: {v}')
        res +=v;
    return res/5


if __name__ == '__main__':
    #train_df = pd.read_feather('train_24cols.feather')
    #train_df = pd.read_csv('train_with_features.csv')
    train_df = pd.read_csv('train_with_features_NO_ST.csv')
    train_df = fill_missing(train_df)
    trn_idx0, val_idx0 = first(GroupKFold().split(train_df, groups = train_df.time_id))
    dls = get_dls(train_df, 100, trn_idx0, val_idx0)

    pruner = optuna.pruners.NopPruner()#optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    sampler = None#optuna.samplers.CmaEsSampler(warn_independent_sampling=False, consider_pruned_trials=False, n_startup_trials=20, restart_strategy='ipop')

    storage = optuna.storages.RDBStorage(
    url='sqlite:///optuna.db',
    engine_kwargs={"connect_args": {"timeout": 10}})

  
    study = optuna.create_study(direction="minimize", study_name = 'parallel_no_st2_parammult', storage=storage, load_if_exists=True, pruner=pruner, sampler=sampler)
    #study.optimize(functools.partial(train, dls=dls),n_trials=500)
    best = study.best_trial
    dlss = [get_dls(train_df,100, trn_idx, val_idx) for trn_idx, val_idx in GroupKFold().split(train_df, groups = train_df.time_id)]
    print('CROSS VALID:' ,train_cross_valid(best, dlss ))

    