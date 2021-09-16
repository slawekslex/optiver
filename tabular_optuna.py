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
    #procs_nn = [Categorify,FillMissing, Normalize]
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(df_train.fillna(0), procs_nn, cat_nn, cont_nn,
                      splits=splits, 
                       
                      y_names='target')

    return to_nn.dataloaders(1024)

def train_network(trial, dls):
    n_layers = trial.suggest_int('n_layers',2,6)
    dropouts = []
    layer_sizes = []
    for i in range(n_layers):
        p = 0 if i==0 else trial.suggest_float(f'p{i}', 0, .5)
        dropouts.append(p)
        layer_sizes.append(trial.suggest_int(f'layer_{i}', 10, 1000))
    embed_p = trial.suggest_float('embed_p', 0, .5)
    config = {'ps': dropouts, 'embed_p':embed_p, 'lin_first':False}
    lr = trial.suggest_float('lr', 1e-3, 2e-2)
    learn = tabular_learner(dls, y_range=(0,.1), layers=layer_sizes, 
                        n_out=1, loss_func = rmspe, metrics=AccumMetric(rmspe), config=config)
    with learn.no_bar():
        with learn.no_logging():
            learn.fit_one_cycle(50, lr)
    last5 = L(learn.recorder.values).itemgot(2)[-5:]
    return np.mean(last5)


def train_cross_valid(trial, dlss):
    res = 0
    for idx, dls in enumerate(dlss):
        v = train_network(trial, dls)
        print(f'fold {idx}: {v}')
        res += v/5
    print('cross valid:', res)
    return res

if __name__ == '__main__':
    df_train = pd.read_csv('train_with_features.csv')
    df_train = df_train.fillna(0)
    print('loading datasets...')
    dlss = [get_dls(df_train, [list(trn_idx), list(val_idx)]) for trn_idx, val_idx in GroupKFold().split(df_train, groups = df_train.time_id)]
    print('creating study..')



    study = optuna.create_study(direction="minimize", study_name = 'tabular_learner_features', storage='sqlite:///optuna.db',load_if_exists=True)

    study.optimize(functools.partial(train_cross_valid, dlss=dlss))

