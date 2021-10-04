# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-09-24T21:02:55.133624Z","iopub.execute_input":"2021-09-24T21:02:55.133952Z","iopub.status.idle":"2021-09-24T21:02:59.459954Z","shell.execute_reply.started":"2021-09-24T21:02:55.133877Z","shell.execute_reply":"2021-09-24T21:02:59.458966Z"}}
import sys
sys.path.append('../usr/lib/optiver_features_private/optiver_features_private.py')
from fastai.tabular.all import *
from multiprocessing import Pool
from sklearn.model_selection import KFold, GroupKFold
import lightgbm as lgb
from optiver_features_private import *

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.461476Z","iopub.execute_input":"2021-09-24T21:02:59.461814Z","iopub.status.idle":"2021-09-24T21:02:59.477519Z","shell.execute_reply.started":"2021-09-24T21:02:59.461778Z","shell.execute_reply":"2021-09-24T21:02:59.476497Z"},"jupyter":{"outputs_hidden":false}}
@delegates(Learner.__init__)
def tabular_learner(dls, layers=None, emb_szs=None, config=None, n_out=None, y_range=None, **kwargs):
    "Get a `Learner` using `dls`, with `metrics`, including a `TabularModel` created using the remaining params."
    if config is None: config = tabular_config()
    if layers is None: layers = [200,100]
    to = dls.train_ds
    emb_szs = get_emb_sz(dls.train_ds, {} if emb_szs is None else emb_szs)
    if n_out is None: n_out = get_c(dls)
    assert n_out, "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"
    if y_range is None and 'y_range' in config: y_range = config.pop('y_range')
    model = TabularModel(emb_szs, len(dls.cont_names), n_out, layers, y_range=y_range, **config)
    return TabularLearner(dls, model, **kwargs)

class TabularModel(Module):
    "Basic model for tabular data."
    def __init__(self, emb_szs, n_cont, out_sz, layers, ps=None, embed_p=0.,
                 y_range=None, use_bn=True, bn_final=False, bn_cont=True, act_cls=nn.ReLU(inplace=True),
                 lin_first=True):
        ps = ifnone(ps, [0]*len(layers))
        if not is_listy(ps): ps = [ps]*len(layers)
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.bn_cont = nn.BatchNorm1d(n_cont) if bn_cont else None
        n_emb = sum(e.embedding_dim for e in self.embeds)
        self.n_emb,self.n_cont = n_emb,n_cont
        sizes = [n_emb + n_cont] + layers + [out_sz]
        actns = [act_cls for _ in range(len(sizes)-2)] + [None]
        _layers = [LinBnDrop(sizes[i], sizes[i+1], bn=use_bn and (i!=len(actns)-1 or bn_final), p=p, act=a, lin_first=lin_first)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        if y_range is not None: _layers.append(SigmoidRange(*y_range))
        self.layers = nn.Sequential(*_layers)

    def forward(self, x_cat, x_cont=None):
        if self.n_emb != 0:
            x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
        if self.n_cont != 0:
            if self.bn_cont is not None: x_cont = self.bn_cont(x_cont)
            x = torch.cat([x, x_cont], 1) if self.n_emb != 0 else x_cont
        return self.layers(x)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.479523Z","iopub.execute_input":"2021-09-24T21:02:59.479923Z","iopub.status.idle":"2021-09-24T21:02:59.489678Z","shell.execute_reply.started":"2021-09-24T21:02:59.479879Z","shell.execute_reply":"2021-09-24T21:02:59.488706Z"},"jupyter":{"outputs_hidden":false}}
data_dir = Path('../input/optiver-realized-volatility-prediction')

# %% [markdown]
# ## Generate 5m datasets

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.491133Z","iopub.execute_input":"2021-09-24T21:02:59.491455Z","iopub.status.idle":"2021-09-24T21:02:59.499964Z","shell.execute_reply.started":"2021-09-24T21:02:59.491418Z","shell.execute_reply":"2021-09-24T21:02:59.499162Z"},"jupyter":{"outputs_hidden":false}}
book_feature_dict = {
    wap1: [],
    wap2: [],
    log_return1: [np.std],
    log_return2: [np.std],
  
    price_spread:[np.mean],
    total_volume:[np.mean],
}
trade_feature_dict = {
        log_return_price: [np.std],
        'seconds_in_bucket':[np.size],
        'size':[np.sum],
        'order_count':[np.sum],
}
time_id_features=[]
time_id_aggregations = []
stock_id_features = []
stock_id_aggregations = [time_id_aggregations]
time_windows = [(0,300), (0,100), (100,200), (200,300)]

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.500961Z","iopub.execute_input":"2021-09-24T21:02:59.501209Z","iopub.status.idle":"2021-09-24T21:02:59.51016Z","shell.execute_reply.started":"2021-09-24T21:02:59.501178Z","shell.execute_reply":"2021-09-24T21:02:59.509184Z"},"jupyter":{"outputs_hidden":false}}
def generate_0_to_5():
    ofg = OptiverFeatureGenerator(book_feature_dict, trade_feature_dict, time_windows, time_id_features,time_id_aggregations, stock_id_features, stock_id_aggregations)
    test_df5m = ofg.generate_test_df()
    ofg_targ =OptiverFeatureGenerator({wap1:[], log_return1:[realized_volatility]}, {'seconds_in_bucket':[np.size]}, [(300,600)], [],[],[],[])
    targ_df = ofg_targ.generate_test_df()
    test_df5m['target'] = targ_df.log_return1_realized_volatility_300_600
    test_df5m.time_id = test_df5m.time_id + 100_000
    
        
    train_df5m = pd.read_feather('../input/optiver-private-data/train_5m.feather')
    concat_df = pd.concat([train_df5m, test_df5m], axis=0)
    concat_df.target = concat_df.target.fillna(1e-4)
    concat_df.target = concat_df.target.replace(0, 1e-4)
    return  concat_df

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.511544Z","iopub.execute_input":"2021-09-24T21:02:59.512081Z","iopub.status.idle":"2021-09-24T21:02:59.521535Z","shell.execute_reply.started":"2021-09-24T21:02:59.512042Z","shell.execute_reply":"2021-09-24T21:02:59.520722Z"},"jupyter":{"outputs_hidden":false}}
def rename_col(x):
    return x.replace('300', '600').replace('200', '500').replace('100', '400').replace('_0', '_300')
def rename_col_back(x):
    return x.replace('300', '0').replace('400', '100').replace('500', '200').replace('600', '300')

def generate_5_to_10():
    tw = [(x+300, y+300) for x,y in time_windows]
    time_id_feat = [rename_col(x) for x in time_id_features]
    stock_id_feat = [rename_col(x) for x in stock_id_features]
    ofg = OptiverFeatureGenerator(book_feature_dict, trade_feature_dict, tw, time_id_feat,time_id_aggregations, stock_id_feat, stock_id_aggregations)

    test_df = ofg.generate_test_df()
    test_df.columns = [rename_col_back(x) for x in test_df.columns]
    test_df.time_id = test_df.time_id + 100_000
    train_df = pd.read_feather('../input/optiver-private-data/train_5_10.feather').drop('target', axis=1)
    concat_df = pd.concat([train_df, test_df], axis=0)
    return concat_df

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:02:59.522744Z","iopub.execute_input":"2021-09-24T21:02:59.523157Z","iopub.status.idle":"2021-09-24T21:03:01.925223Z","shell.execute_reply.started":"2021-09-24T21:02:59.523117Z","shell.execute_reply":"2021-09-24T21:03:01.923941Z"},"jupyter":{"outputs_hidden":false}}
train_df5m =  generate_0_to_5()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:03:01.929401Z","iopub.execute_input":"2021-09-24T21:03:01.929702Z","iopub.status.idle":"2021-09-24T21:03:02.015524Z","shell.execute_reply.started":"2021-09-24T21:03:01.92967Z","shell.execute_reply":"2021-09-24T21:03:02.014412Z"},"jupyter":{"outputs_hidden":false}}
cols_to_keep = ['log_return2_std_0_300',
 
 'stock_id','row_id', 'time_id', 'target',
 'log_return_price_std_0_300',
  'order_count_sum_0_300',
 'seconds_in_bucket_size_0_300',
 'size_sum_0_300',
 'log_return1_std_0_300',
 'log_return1_std_100_200',
 'log_return1_std_200_300',
 'price_spread_mean_0_300',

 'total_volume_mean_0_300']
train_df5m = train_df5m[cols_to_keep]

# %% [markdown]
# ## Train the 5m model

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:03:02.017624Z","iopub.execute_input":"2021-09-24T21:03:02.018059Z","iopub.status.idle":"2021-09-24T21:03:02.031301Z","shell.execute_reply.started":"2021-09-24T21:03:02.018019Z","shell.execute_reply":"2021-09-24T21:03:02.03052Z"},"jupyter":{"outputs_hidden":false}}
cont_nn,cat_nn = cont_cat_split(train_df5m, max_card=9000, dep_var=['target'])
cat_nn = ['time_id', 'stock_id']

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:03:02.032613Z","iopub.execute_input":"2021-09-24T21:03:02.033007Z","iopub.status.idle":"2021-09-24T21:03:15.317667Z","shell.execute_reply.started":"2021-09-24T21:03:02.03297Z","shell.execute_reply":"2021-09-24T21:03:15.316817Z"},"jupyter":{"outputs_hidden":false}}
procs_nn = [Categorify, FillMissing,Normalize]
splits = RandomSplitter(seed=3, valid_pct=.1)(train_df5m)
dls = TabularPandas(train_df5m, procs_nn, cat_nn, cont_nn,
                      splits=splits, 
                      y_names='target').dataloaders(1024)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:03:15.31894Z","iopub.execute_input":"2021-09-24T21:03:15.319271Z","iopub.status.idle":"2021-09-24T21:03:15.35314Z","shell.execute_reply.started":"2021-09-24T21:03:15.319237Z","shell.execute_reply":"2021-09-24T21:03:15.352432Z"},"jupyter":{"outputs_hidden":false}}
def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()


config={ 'ps':[0,0,0], 'embed_p':0.25 }
learn = tabular_learner(dls, y_range=(0,.1), layers=[200,100,40], emb_szs={'stock_id':5, 'time_id':10}, 
                        n_out=1, loss_func = rmspe, metrics=AccumMetric(rmspe), config=config)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:03:15.354729Z","iopub.execute_input":"2021-09-24T21:03:15.354966Z","iopub.status.idle":"2021-09-24T21:06:57.586455Z","shell.execute_reply.started":"2021-09-24T21:03:15.354943Z","shell.execute_reply":"2021-09-24T21:06:57.585521Z"},"jupyter":{"outputs_hidden":false}}
learn.fit_one_cycle(30, 5e-3)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:06:57.587779Z","iopub.execute_input":"2021-09-24T21:06:57.588126Z","iopub.status.idle":"2021-09-24T21:06:57.593743Z","shell.execute_reply.started":"2021-09-24T21:06:57.588088Z","shell.execute_reply":"2021-09-24T21:06:57.592783Z"},"jupyter":{"outputs_hidden":false}}
categorify = dls.procs[2]
time_embs = learn.model.embeds[0].weight.data.cpu()
stock_embs = learn.model.embeds[1].weight.data.cpu()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:06:57.595064Z","iopub.execute_input":"2021-09-24T21:06:57.595587Z","iopub.status.idle":"2021-09-24T21:07:09.706818Z","shell.execute_reply.started":"2021-09-24T21:06:57.595545Z","shell.execute_reply":"2021-09-24T21:07:09.705687Z"},"jupyter":{"outputs_hidden":false}}
test_df = generate_5_to_10()
test_df = test_df[[c for c in cols_to_keep if c != 'target']]
test_dl = dls.test_dl(test_df)
preds, _ = learn.get_preds(dl=test_dl)
preds5_10 = preds.view(-1).numpy()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:09.708623Z","iopub.execute_input":"2021-09-24T21:07:09.709041Z","iopub.status.idle":"2021-09-24T21:07:09.743012Z","shell.execute_reply.started":"2021-09-24T21:07:09.708998Z","shell.execute_reply":"2021-09-24T21:07:09.74219Z"},"jupyter":{"outputs_hidden":false}}
del dls, learn, train_df5m, test_df, test_dl

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:09.744281Z","iopub.execute_input":"2021-09-24T21:07:09.744756Z","iopub.status.idle":"2021-09-24T21:07:09.75456Z","shell.execute_reply.started":"2021-09-24T21:07:09.74472Z","shell.execute_reply":"2021-09-24T21:07:09.753794Z"},"jupyter":{"outputs_hidden":false}}
def append_embs(data_df, categorify, time_embs, stock_embs):
    time_id_embs = dict()
    for idx, time_id in enumerate(categorify.classes['time_id']):
        time_id_embs[time_id] = time_embs[idx].tolist()

    stock_id_embs = dict()
    for idx, stock_id in enumerate(categorify.classes['stock_id']):
        stock_id_embs[stock_id] = stock_embs[idx].tolist()

    all_embs = []
    for _, row in (data_df[['stock_id', 'time_id']].iterrows()):
        emb1 = stock_id_embs[row.stock_id]
        emb2 = time_id_embs[row.time_id]
        all_embs.append(emb1+emb2)
    columns = [f'stock_emb{i}' for i in range(stock_embs.shape[1])] +[f'time_emb{i}' for i in range(time_embs.shape[1])]
    embs_df = pd.DataFrame(all_embs, columns=columns)

    return pd.concat([data_df, embs_df], axis=1)

# %% [markdown]
# ## Train 10m LGB

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:09.755771Z","iopub.execute_input":"2021-09-24T21:07:09.756218Z","iopub.status.idle":"2021-09-24T21:07:09.771287Z","shell.execute_reply.started":"2021-09-24T21:07:09.756179Z","shell.execute_reply":"2021-09-24T21:07:09.770356Z"},"jupyter":{"outputs_hidden":false}}
def rmspe_np(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe_np(y_true, y_pred), False

def train_LGB(train):
    # Hyperparammeters (optimized)
    seed = 29
    params = {
        'learning_rate': 0.1,        
        'lambda_l1': 2,
        'lambda_l2': 7,
        'num_leaves': 800,
        'min_sum_hessian_in_leaf': 20,
        'feature_fraction': 0.8,
        'feature_fraction_bynode': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 42,
        'min_data_in_leaf': 700,
        'max_depth': 4,
        'seed': seed,
        'feature_fraction_seed': seed,
        'bagging_seed': seed,
        'drop_seed': seed,
        'data_random_seed': seed,
        'objective': 'rmse',
        'boosting': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
       # 'device':'gpu'
    }   
    
    # Split features and target
    x = train.drop(['row_id', 'target', 'time_id'], axis = 1)
    y = train['target']
    # Transform stock id to a numeric value
    x['stock_id'] = x['stock_id'].astype(int)
    models =[]
    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])
    # Create a KFold object
    kfold = GroupKFold()
    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x, groups = train.time_id)):
        print(f'Training fold {fold + 1}')
        x_train, x_val = x.iloc[trn_ind], x.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights, categorical_feature = ['stock_id'])
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights, categorical_feature = ['stock_id'])
        model = lgb.train(params = params, 
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          num_boost_round = 3000, 
                          early_stopping_rounds = 25, 
                          verbose_eval = 100,
                          feval = feval_rmspe)
        models.append(model)
        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)
        # Predict the test set
        #test_predictions += model.predict(x_test) / 10
        
    rmspe_score = rmspe_np(y, oof_predictions)
    print(f'Our out of folds RMSPE is {rmspe_score}')
    # Return test predictions
    return models

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:29:23.949852Z","iopub.execute_input":"2021-09-24T21:29:23.950235Z","iopub.status.idle":"2021-09-24T21:29:23.963538Z","shell.execute_reply.started":"2021-09-24T21:29:23.950193Z","shell.execute_reply":"2021-09-24T21:29:23.96214Z"},"jupyter":{"outputs_hidden":false}}
def train_one_nn(train_df, test_df, split):
    cont_nn,cat_nn = cont_cat_split(train_df, max_card=9000, dep_var='target')
    cat_nn = []

    
    procs_nn = [Categorify,Normalize]
    dls = TabularPandas(train_df, procs_nn, cat_nn, cont_nn,
                          splits=split, y_names='target').dataloaders(1024)

 

    config={'lin_first':False, 'embed_p':.1, 'ps':[0,.4,0]}
    learn = tabular_learner(dls, y_range=(0,.1), layers=[400,200,100], 
                        n_out=1, 
                        loss_func = rmspe,opt_func=ranger, 
                        metrics=AccumMetric(rmspe), config=config)
    learn.fit_flat_cos(20, 5e-3, wd=.2)
    val_score = learn.recorder.metrics[0].value
    test_dl = dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)
#     score =rmspe_np(test_df.target, preds.view(-1).numpy())
#     print(score)
    return preds, val_score

def train_nn(train_df, test_df):
    splits = GroupKFold().split(train_df, groups = train_df.time_id)

    if len(test_df) == 3: 
        test_df = test_df.fillna(0)

    preds=[]
    oof =0
    for trn_idx, val_idx in splits:
        p, val_score = train_one_nn(train_df, test_df, [list(trn_idx), list(val_idx)])
        oof += val_score/5
        preds.append(p)
    preds =torch.cat(preds, dim=1)
    torch.save(preds, 'five_minutes.pt')
    print('OOF', oof)
    return preds.median(dim=1)[0].numpy()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:09:45.97221Z","iopub.execute_input":"2021-09-24T21:09:45.972537Z","iopub.status.idle":"2021-09-24T21:09:47.028012Z","shell.execute_reply.started":"2021-09-24T21:09:45.972507Z","shell.execute_reply":"2021-09-24T21:09:47.027106Z"},"jupyter":{"outputs_hidden":false}}
train_10m = pd.read_feather('../input/optiver-private-data/train_351cols.feather')
to_drop = [c for c in train_10m.columns.to_list() if c.endswith('_time') or c.endswith('_stock')] 
train_10m = train_10m.drop(to_drop, axis=1)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:10:24.611758Z","iopub.execute_input":"2021-09-24T21:10:24.612103Z","iopub.status.idle":"2021-09-24T21:11:04.707878Z","shell.execute_reply.started":"2021-09-24T21:10:24.612075Z","shell.execute_reply":"2021-09-24T21:11:04.706975Z"},"jupyter":{"outputs_hidden":false}}
train_10m = append_embs(train_10m, categorify, time_embs, stock_embs)

train_10m['5m_pred']= preds5_10[:len(train_10m)]

# %% [markdown]
# ## Get 10m test predictions

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:12:19.681719Z","iopub.execute_input":"2021-09-24T21:12:19.682207Z","iopub.status.idle":"2021-09-24T21:12:19.691067Z","shell.execute_reply.started":"2021-09-24T21:12:19.682172Z","shell.execute_reply":"2021-09-24T21:12:19.690173Z"},"jupyter":{"outputs_hidden":false}}
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
agg_cols = ['log_return_price_std', 'log_return1_std', 'log_return2_std', 'size_sum', 'order_count_sum']
time_id_features = [f'{col}_{x}_{y}' for x,y in time_windows for col in agg_cols] 
time_id_aggregations = ['mean', 'std', 'min' ]
stock_id_features = time_id_features
stock_id_aggregations = time_id_aggregations
ofg = OptiverFeatureGenerator(book_feature_dict, trade_feature_dict, time_windows, time_id_features,time_id_aggregations, stock_id_features, stock_id_aggregations)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:12:26.358075Z","iopub.execute_input":"2021-09-24T21:12:26.358419Z","iopub.status.idle":"2021-09-24T21:12:27.03298Z","shell.execute_reply.started":"2021-09-24T21:12:26.358386Z","shell.execute_reply":"2021-09-24T21:12:27.031745Z"},"jupyter":{"outputs_hidden":false}}
test_10m = ofg.generate_test_df()
test_10m.time_id = test_10m.time_id + 100_000
test_10m = test_10m.drop(to_drop, axis=1)
test_10m = append_embs(test_10m, categorify, time_embs, stock_embs)
test_10m['5m_pred'] = preds5_10[len(train_10m):]

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:15.5876Z","iopub.status.idle":"2021-09-24T21:07:15.588161Z"},"jupyter":{"outputs_hidden":false}}
def pred_lgb(test_df, models):
    test_df = test_df.drop(['row_id', 'time_id'], axis=1)
    res = np.zeros(len(test_df))
    for model in models:
        preds = model.predict(test_df)
        res += preds
    return res / len(models)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:13:18.96608Z","iopub.execute_input":"2021-09-24T21:13:18.966472Z","iopub.status.idle":"2021-09-24T21:13:19.319732Z","shell.execute_reply.started":"2021-09-24T21:13:18.966429Z","shell.execute_reply":"2021-09-24T21:13:19.318744Z"},"jupyter":{"outputs_hidden":false}}
train_10m = train_10m.drop('stock_id', axis=1)
test_10m = test_10m.drop('stock_id', axis=1)
train_10m, test_10m = train_10m.fillna(0), test_10m.fillna(0)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:13:38.831837Z","iopub.execute_input":"2021-09-24T21:13:38.832197Z","iopub.status.idle":"2021-09-24T21:13:39.055845Z","shell.execute_reply.started":"2021-09-24T21:13:38.832167Z","shell.execute_reply":"2021-09-24T21:13:39.054842Z"},"jupyter":{"outputs_hidden":false}}
import gc
gc.collect()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:13:43.36934Z","iopub.execute_input":"2021-09-24T21:13:43.369689Z","iopub.status.idle":"2021-09-24T21:28:00.751396Z","shell.execute_reply.started":"2021-09-24T21:13:43.369652Z","shell.execute_reply":"2021-09-24T21:28:00.747019Z"},"jupyter":{"outputs_hidden":false}}
nn_preds = train_nn(train_10m, test_10m)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:15.591127Z","iopub.status.idle":"2021-09-24T21:07:15.591716Z"},"jupyter":{"outputs_hidden":false}}
# models = train_LGB(train_10m)
# for m in models[:2]:
#     lgb.plot_importance(m, max_num_features=20, importance_type='gain')
# lgb_preds = pred_lgb(test_10m, models)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-24T21:07:15.592826Z","iopub.status.idle":"2021-09-24T21:07:15.593365Z"},"jupyter":{"outputs_hidden":false}}


test_10m['target']=nn_preds
fname = 'five_minutes.feather'
test_10m[['row_id', 'target']].reset_index().to_feather(fname)