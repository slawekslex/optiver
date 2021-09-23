from fastai.tabular.all import *
from sklearn.model_selection import KFold, GroupKFold
import optuna
from optuna.integration import FastAIPruningCallback
import sqlalchemy

STOCK_COUNT = 112


def fill_missing(train_df):
    
    all_times = train_df.time_id.unique()
    all_stocks = train_df.stock_id.unique()
    filled_df = train_df.copy()
    filled_df=filled_df.set_index(['time_id', 'stock_id'])
    new_index = pd.MultiIndex.from_product([all_times, all_stocks], names = ['time_id', 'stock_id'])
    filled_df = filled_df.reindex(new_index).reset_index()
    filled_df = filled_df.fillna(0)
    return filled_df

def subtract_windows(df, time_windows):
    for s,e in time_windows[1:]:
        for c in df.columns:
            wind = f'{s}_{e}'
            if c.endswith(wind): 
                pref = c[:-len(wind)]
                main_col = pref+'0_600'
                df[c] = df[main_col]-df[c]
    return df

def append_trade_count(train_df, time_windows):
    for s,e in time_windows:
        train_df[f'number_trades_{s}_{e}'] = 'more'
        for val in range(3): train_df.loc[train_df[f'seconds_in_bucket_size_{s}_{e}']==val, f'number_trades_{s}_{e}'] = val
    return train_df

def tauify(train_df):
    for c in train_df.columns:
        if 'sum' in c: train_df[c] = np.sqrt(1/(train_df[c]+1))
    return train_df

def post_process(train_df, time_windows, do_subtract, do_append, do_tau):
    train_df = fill_missing(train_df)
    if do_subtract: train_df = subtract_windows(train_df, time_windows)
    if do_append: train_df = append_trade_count(train_df, time_windows)
    if do_tau: train_df = tauify(train_df)
    return train_df

    

class Jitter(ItemTransform):
    def __init__(self, jit_std):
            super().__init__()
            self.split_idx = 0
            self.jit_std = jit_std
            
    def encodes(self, b):
        #print('doing jitter ', self.jit_std)
        jitter = torch.empty_like(b[1]).normal_(0, self.jit_std)
        b[1] += jitter
        return b

class MaskTfm(ItemTransform):
    
    def __init__(self, mask_perc):
        super().__init__()
        self.split_idx = 0
        self.mask_perc = mask_perc
    
    def mask(self, x, indices):
        x[torch.tensor(indices, device=x.device)] = 0
        return x
    
    def encodes(self, x):
        #print('doing mask', self.mask_perc)
        n = len(x[0])
        to_mask = (n * self.mask_perc) // 100
        indices = np.random.choice(np.array(range(n)), to_mask, replace=False)
        x = [self.mask(y, indices) for y in x]
        
        return x

class MyDataLoader(TabDataLoader):
    def __init__(self, dataset, jit_std, mask_perc, bs=16, shuffle=False, after_batch=None, num_workers=0,  **kwargs):
        if after_batch is None: after_batch = L(TransformBlock().batch_tfms)+ReadTabBatch(dataset) + [Jitter(jit_std), MaskTfm(mask_perc)]
        super().__init__(dataset, bs=bs, shuffle=shuffle, after_batch=after_batch, num_workers=num_workers, **kwargs)

    def shuffle_fn(self, idxs):
        idxs = np.array(idxs).reshape(-1,112)
        np.random.shuffle(idxs)
        return idxs.reshape(-1).tolist()

def get_dls(train_df, bs, trn_idx, val_idx, jit_std=.13, mask_perc=8):
    cont_nn,cat_nn = cont_cat_split(train_df, max_card=9000, dep_var='target')
    cat_nn=[x for x in cat_nn if not x in ['row_id', 'time_id']]
    
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(train_df, procs_nn, cat_nn, cont_nn, splits=[list(trn_idx), list(val_idx)], y_names='target')
    dls = to_nn.dataloaders(bs=STOCK_COUNT * bs, shuffle=True, dl_type = MyDataLoader, jit_std=jit_std, mask_perc=mask_perc)
    dls.train_ds.split_idx=0
    dls.valid_ds.split_idx=1
    return dls


class TimeEncoding(nn.Module):
    def __init__(self, inp_size, bottleneck, p, multiplier):
        super().__init__()
        self.multiplier  = multiplier#nn.Parameter(torch.tensor(multiplier)) 
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

class ParallelBlock(nn.Module):
    def __init__(self, block_size, p, time_p, bottleneck, multiplier, do_skip):
        super().__init__()
        self.do_skip = do_skip
        self.layers = nn.Sequential(
            nn.Linear(block_size, block_size),
            BN(block_size ),
            nn.Dropout(p),
            nn.ReLU(True),
            TimeEncoding(block_size, bottleneck, time_p, multiplier)
        )
    def forward(self, x):
        y = self.layers(x)
        if self.do_skip: return (y + x) /2
        else: return y
    
class ParallelModel(nn.Module):
    def __init__(self, inp_size, emb_szs, block_size, ps, bottleneck, time_ps, multipliers, embed_p, do_skip ):
        super().__init__()
        
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.embed_drop = nn.Dropout(embed_p)
        n_emb = sum(e.embedding_dim for e in self.embeds)
        
        layers = [nn.Linear(inp_size+n_emb, block_size),
                 BN(block_size),
                 nn.ReLU(True)]
        for p, time_p, multiplier in zip( ps, time_ps, multipliers):            
            layers.append(ParallelBlock(block_size, p, time_p, bottleneck, multiplier, do_skip))
            
        layers.append(nn.Linear(block_size, 1))
        layers.append(SigmoidRange(0, .1))
        self.layers = nn.Sequential(*layers)
    
    
    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.embed_drop(x)
        x = torch.cat([x_cont, x], dim=1)
        return self.layers(x)
    
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

def train(trial, train_df, trn_idx, val_idx, save_as=None):
    do_subtract = False#trial.suggest_categorical('do_subtract', [True, False])
    do_append = False#trial.suggest_categorical('do_append', [True, False])
    do_tau = True#trial.suggest_categorical('do_tau', [True, False])
    train_df = post_process(train_df, time_windows, do_subtract, do_append, do_tau)
    
    jit_std=trial.suggest_float('jit_std', 0, .1)
    mask_perc=trial.suggest_int('mask_perc', 5, 20)
    
    
    if trn_idx is None:
        trn_idx, val_idx = first(GroupKFold().split(train_df, groups = train_df.time_id))
    dls = get_dls(train_df, 100, trn_idx, val_idx, jit_std=jit_std, mask_perc = mask_perc)
    inp_size = len(dls.cont_names)
    
    do_skip =True# trial.suggest_categorical('do_skip', [True, False])
    emb_size = trial.suggest_int('emb_size', 3, 30)
    emb_sizes = [(len(c_vals), emb_size if c_name == 'stock_id' else 3) for c_name, c_vals in dls.train.classes.items()]
    emb_p = trial.suggest_float(f'emb_p', 0, .5)
    block_size = trial.suggest_int(f'block_size', 50, 1000) 
    ps = [trial.suggest_float(f'p{i}', 0, .8) for i in range(4)]
    bottleneck = trial.suggest_int('bottleneck', 5, 100)
    time_ps = [trial.suggest_float(f'time_p{i}', 0, .5) for i in range(4)]
    multipliers = [trial.suggest_float(f'multiplier{i}', .01, .5) for i in range(4)]
    
    lr = float(trial.suggest_float('lr', 1e-3, 1e-2))
    wd = float(trial.suggest_float('wd', 0, .2))
    
    
    model = ParallelModel(inp_size, emb_sizes, block_size, ps, bottleneck, time_ps, multipliers, emb_p, do_skip)
    learn = Learner(dls,model = model, loss_func=rmspe, metrics=AccumMetric(rmspe), opt_func=ranger,
        cbs = FastAIPruningCallback(trial, 'rmspe'), wd=wd).to_fp16()
    # with learn.no_bar():
    #     with learn.no_logging():    
    learn.fit_flat_cos(50, lr)
    if save_as:
        learn.save(save_as)
    last5 = L(learn.recorder.values).itemgot(2)[-5:]
    return np.mean(last5)

def train_cross_valid(trial, train_df, save_as=None):
    res = 0
    splits = GroupKFold().split(train_df, groups = train_df.time_id)
    for idx, (trn_idx, val_idx) in enumerate(splits):
        v = train(trial, train_df, trn_idx, val_idx, save_as + str(idx) if save_as else None)
        print(f'fold {idx}: {v}')
        res +=v;
    return res/5


if __name__ == '__main__':
    #train_df = pd.read_feather('train_24cols.feather')
    #train_df = pd.read_csv('train_with_features.csv')
    time_windows = [(0,600), (0,100), (100,200), (200,300), (300,400), (400, 500), (500,600)]
    train_df = pd.read_feather('train_141cols.feather')
    
    

    pruner = optuna.pruners.NopPruner()
    sampler = None#optuna.samplers.CmaEsSampler(warn_independent_sampling=False, consider_pruned_trials=False, n_startup_trials=20, restart_strategy='ipop')

    storage = optuna.storages.RDBStorage(
    url='sqlite:///optuna.db',
    engine_kwargs={"connect_args": {"timeout": 10}})

  
    study = optuna.create_study(direction="minimize", study_name = 'four_blocks', storage=storage, load_if_exists=True, pruner=pruner, sampler=sampler)
    study.optimize(functools.partial(train, train_df=train_df, trn_idx=None, val_idx=None),n_trials=500)
    # best = study.best_trial
    # dlss = [get_dls(train_df,100, trn_idx, val_idx) for trn_idx, val_idx in GroupKFold().split(train_df, groups = train_df.time_id)]
    # print('CROSS VALID:' ,train_cross_valid(best, dlss ))

    