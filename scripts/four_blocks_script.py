import sys
sys.path.append('../usr/lib/optiver_features_private/optiver_features_private.py')
from optiver_features_private import *
from fastai.tabular.all import *
from sklearn.model_selection import KFold, GroupKFold
import gc

STOCK_COUNT = 112

params_4B = {
 'do_append': False,
 'do_skip': True,
 'do_subtract': False,
 'do_tau': True,
 'block_size': 566,
 'bottleneck': 89,
 'emb_p': 0.22892755376131763,
 'emb_size': 29,
 'jit_std': 0.044428121554388224,
 'lr': 0.0090050544233066,
 'mask_perc': 15,
 'multiplier0': 0.36610514833346075,
 'multiplier1': 0.4699461659624539,
 'multiplier2': 0.19267216205111673,
 'multiplier3': 0.1103899309385393,
 'p0': 0.7779388520646903,
 'p1': 0.4268044282083482,
 'p2': 0.017189932979278854,
 'p3': 0.6956976227194849,
 'time_p0': 0.040130990403726904,
 'time_p1': 0.2795015767348503,
 'time_p2': 0.2534561744168111,
 'time_p3': 0.35356055022302363,
 'wd': 0.1755624424764211
}

def fill_missing(train_df, all_stocks):
    all_times = train_df.time_id.unique()
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

def post_process(train_df, all_stocks, time_windows, do_subtract, do_append, do_tau):
    train_df = fill_missing(train_df, all_stocks)
    if do_subtract: train_df = subtract_windows(train_df, time_windows)
    if do_append: train_df = append_trade_count(train_df, time_windows)
    if do_tau: train_df = tauify(train_df)
    return train_df

def generate_train_test_parallel(params):
    book_feature_dict = {
        wap1: [np.mean, np.std, 'nunique'],
        wap2: [np.mean, np.std],
        log_return1: [np.std],
        log_return2: [np.std],
        ask_spread: [np.mean, np.std],
        price_spread:[np.mean, np.std],
        total_volume:[np.mean, np.std],
    }
    trade_feature_dict = {
            log_return_price: [np.std, np.mean],
            'seconds_in_bucket':[np.size],
            'size':[np.sum],
            'order_count':[np.sum],
    }
    time_windows = [(0,600), (0,100), (100,200), (200,300), (300,400), (400, 500), (500,600)]
    ofg = OptiverFeatureGenerator(book_feature_dict, trade_feature_dict, time_windows)
    train_df = pd.read_feather('../input/optiver-private-data/train_141cols.feather')  
    test_df = ofg.generate_test_df()
    original_rows = test_df.row_id
    all_stocks = train_df.stock_id.unique()
    train_df, test_df = (post_process(df, all_stocks, time_windows, params['do_subtract'], params['do_append'], params['do_tau']) for df in (train_df, test_df))
    return train_df, test_df, original_rows

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

class Jitter(ItemTransform):
    def __init__(self, jit_std):
            super().__init__()
            self.split_idx = 0
            self.jit_std = jit_std
            
    def encodes(self, b):
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
    dls = to_nn.dataloaders(bs=112*100, shuffle=True, dl_type = MyDataLoader, jit_std=jit_std, mask_perc=mask_perc)
    dls.train_ds.split_idx=0
    dls.valid_ds.split_idx=1
    return dls

def rmspe_masked(preds, targs):
    mask = targs != 0
    targs, preds = torch.masked_select(targs, mask), torch.masked_select(preds, mask)
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()
def rmspe(preds, targs):

    x = (targs-preds)/targs
    return (x**2).mean().sqrt()

def four_blocks_predictions():
    params = params_4B
    train_df, test_df, original_rows = generate_train_test_parallel(params)
    trn_idx0, val_idx0 = first(GroupKFold().split(train_df, groups = train_df.time_id))
    dls = get_dls(train_df, 100, trn_idx0, val_idx0, params['jit_std'], params['mask_perc'])
    inp_size = len(dls.cont_names)
    do_skip = params['do_skip']
    emb_size = params['emb_size']
    emb_sizes = [(len(c_vals), emb_size if c_name == 'stock_id' else 3) for c_name, c_vals in dls.train.classes.items()]
    block_size = params['block_size']
    ps = [params[f'p{i}'] for i in range(4)]
    bottleneck = params['bottleneck']
    time_ps = [params[f'time_p{i}'] for i in range(4)]
    multipliers = [params[f'multiplier{i}'] for i in range(4)]
    emb_p = params['emb_p']
    model = ParallelModel(inp_size, emb_sizes, block_size, ps, bottleneck, time_ps, multipliers, emb_p, do_skip)
    
    # sanity check
    learn = Learner(dls,model = model, path = '../input/optiver-models-private/', loss_func = rmspe_masked,metrics=AccumMetric(rmspe_masked), opt_func=ranger ).to_fp16()
    learn.load('four_blocks_0')
    preds, ys = learn.get_preds(dl = dls.valid, reorder=False)
    print('should be .2149', rmspe_masked(preds, ys))
    
    res = []
    splits = GroupKFold().split(train_df, groups = train_df.time_id)
    for idx, (trn_idx, val_idx) in enumerate(splits):
        dls = get_dls(train_df, 100, trn_idx, val_idx, params['jit_std'], params['mask_perc'])
        learn = Learner(dls,model = model, path = '../input/optiver-models-private/', loss_func = rmspe_masked,metrics=AccumMetric(rmspe_masked), opt_func=ranger ).to_fp16()
        learn.load(f'four_blocks_{idx}')
        test_dl = dls.test_dl(test_df, jit_std=0, mask_perc=0)
        preds, _ = learn.get_preds(dl=test_dl, reorder=False)
        res.append(preds)
    res = torch.cat(res, dim=1)
    torch.save(res, 'four_blocks.pt')
    res = res.mean(dim=1).numpy()
    
    test_df['target'] = res
    test_df=test_df.loc[test_df.row_id.isin( original_rows)]
    fname = 'four_blocks.feather'
    test_df[['row_id', 'target']].reset_index().to_feather(fname)
   
    return fname

four_blocks_predictions()