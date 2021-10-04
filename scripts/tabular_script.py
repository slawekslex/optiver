import sys
sys.path.append('../usr/lib/optiver_features_private/optiver_features_private.py')
from optiver_features_private import *

from fastai.tabular.all import *
from sklearn.model_selection import KFold, GroupKFold

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

def postpro(df):
    df = df.fillna(0)
    for c in ['wap1_nunique_0_600','wap1_nunique_0_100','wap1_nunique_100_200','wap1_nunique_200_300','wap1_nunique_300_400','wap1_nunique_400_500','wap1_nunique_500_600']:
        df[c] = df[c].astype(np.float32)
    return df

train_df = pd.read_feather('../input/optiver-private-data/train_351cols.feather')
test_df = ofg.generate_test_df()
train_df, test_df = postpro(train_df), postpro(test_df)


trn_idx0, val_idx0 = first(GroupKFold().split(train_df, groups = train_df.time_id))

def get_dls(train_df, trn_idx, val_idx):
    procs_nn = [Categorify, Normalize]
    cont_nn,cat_nn = cont_cat_split(train_df, max_card=9000, dep_var='target')
    cat_nn.remove('time_id'), cat_nn.remove('row_id')

    return TabularPandas(train_df, procs_nn, cat_nn, cont_nn,
                          splits=[list(trn_idx), list(val_idx)], y_names='target').dataloaders(1024)

def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()


dls0 = get_dls(train_df, trn_idx0, val_idx0)
config={'embed_p':.1, 'ps':[0,.4,.1]}
learn = tabular_learner(dls0, y_range=(0,.1), layers=[800,400,200], config = config, path = '../input/optiver-models-private/',
                    n_out=1, loss_func=rmspe, metrics=AccumMetric(rmspe),opt_func=ranger)

learn.load('tuned_0')
preds, ys = learn.get_preds()
print('should be .2209', rmspe(preds, ys))

res = []
splits = GroupKFold().split(train_df, groups = train_df.time_id)
# learn = tabular_learner(dls0, y_range=(0,.1), layers=[800,400,200], config = config, path = '../input/optiver-models-private/',
#                     n_out=1, loss_func=rmspe, metrics=AccumMetric(rmspe),opt_func=ranger)
for idx, (trn_idx, val_idx) in enumerate(splits):
    dls = dls0 #get_dls(train_df, trn_idx, val_idx)
    
    learn.load(f'tuned_{idx}')
    test_dl = dls.test_dl(test_df)
    preds, _ = learn.get_preds(dl=test_dl)
    res.append(preds)

res = torch.cat(res, dim=1)
torch.save(res, 'tabular.pt')
res = res.mean(dim=1).numpy()
test_df['target']=res
fname = 'tabular.feather'
test_df[['row_id', 'target']].reset_index().to_feather(fname)
