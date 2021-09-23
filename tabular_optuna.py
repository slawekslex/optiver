import optuna
from optuna.integration import FastAIPruningCallback
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from optiver_features import *


def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()





def get_dls(df_train, splits):
    cont_nn,cat_nn = cont_cat_split(df_train, max_card=9000, dep_var='target')
    cat_nn.remove('time_id'), cat_nn.remove('row_id')
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(df_train, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names='target')

    return to_nn.dataloaders(1024)



def train_network(trial, dls, train_epochs=30, save_as=None):

    dropouts = []
    layer_sizes = [800,400,200]
    for i in range(3):
        p = 0 if i==0 else trial.suggest_float(f'p{i}', 0, .5)
        dropouts.append(p)
       
    stock_embed_size = trial.suggest_int('stock_emb', 3, 30)
    embed_p = trial.suggest_float('embed_p', 0, .5)
    config = {'ps': dropouts, 'embed_p':embed_p, 'lin_first':False}
    lr = trial.suggest_float('lr', 1e-3, 5e-2)
    learn = tabular_learner(dls, y_range=(0,.1), layers=layer_sizes, emb_szs={'stock_id':stock_embed_size},
                        n_out=1, loss_func = rmspe, metrics=AccumMetric(rmspe), config=config, opt_func = ranger)
    
    
   
    learn.fit_flat_cos(train_epochs, lr)
    if save_as:
        learn.save(save_as)
    last3 = L(learn.recorder.values).itemgot(2)[-3:]
    return np.mean(last3)


def train_cross_valid(trial, dlss):
    res = 0
  
    optimizer = ranger


    
    train_epochs = 30
    bs = 1024
 
    for idx, dls in enumerate(dlss):
        dls.train.bs=bs
        v = train_network(trial, dls, train_epochs)
        print(f'fold {idx}: {v}')
        trial.report(v, step = idx)
        if trial.should_prune():
            raise optuna.TrialPruned(f'Trial was pruned after fold {idx}')
        res += v/5

    print('cross valid:', res)
    return res

if __name__ == '__main__':
    df_train =pd.read_feather('train_351cols.feather')
    df_train = df_train.fillna(0)
    for c in ['wap1_nunique_0_600','wap1_nunique_0_100','wap1_nunique_100_200','wap1_nunique_200_300','wap1_nunique_300_400','wap1_nunique_400_500','wap1_nunique_500_600']:
        df_train[c] = df_train[c].astype(np.float32)
    print('loading datasets...')
    #dlss = [get_dls(df_train, [list(trn_idx), list(val_idx)]) for trn_idx, val_idx in GroupKFold().split(df_train, groups = df_train.time_id)]
    trn_idx, val_idx = first(GroupKFold().split(df_train, groups = df_train.time_id))
    dls = get_dls(df_train, [list(trn_idx), list(val_idx)])
    print('creating study..')


    pruner =  optuna.pruners.NopPruner()
    study = optuna.create_study(direction="minimize", study_name = 'tabular_v2', storage='sqlite:///optuna.db',load_if_exists=True, pruner=pruner)

    #study.optimize(functools.partial(train_network, dls=dls))
    best_trial = study.best_trial
    print(best_trial)
    dlss = [get_dls(df_train, [list(trn_idx), list(val_idx)]) for trn_idx, val_idx in GroupKFold().split(df_train, groups = df_train.time_id)]
    train_cross_valid(best_trial, dlss)

