import optuna
from optuna.integration import FastAIPruningCallback
from fastai.tabular.all import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, GroupKFold
from optiver_features import *


def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()


def train_network(trial, dls):
    n_layers = trial.suggest_int('n_layers',2,5)
    dropouts = []
    layer_sizes = []
    for i in range(n_layers):
        dropouts.append(trial.suggest_float(f'p{i}', 0, .5))
        layer_sizes.append(trial.suggest_int(f'layer_{i}', 100, 1000))
    embed_p = trial.suggest_float('embed_p', 0, .5)
    config = {'ps': dropouts, 'embed_p':embed_p}
    lr = trial.suggest_float('lr', 1e-3, 2e-2, log=True)
    learn = tabular_learner(dls, y_range=(0,.1), layers=layer_sizes, 
                        n_out=1, loss_func = rmspe, metrics=AccumMetric(rmspe), config=config, cbs=FastAIPruningCallback(trial, monitor="rmspe"))
    #with learn.no_bar():
    with learn.no_logging():
        learn.fit_one_cycle(50, lr)
    return learn.recorder.metrics[0].value

if __name__ == '__main__':
    df_train = pd.read_csv('train_with_features_NO_ST.csv')
    trn_idx, val_idx = first(GroupKFold().split(df_train, groups = df_train.time_id))
    splits=[list(trn_idx), list(val_idx)]
    cont_nn,cat_nn = cont_cat_split(df_train, max_card=9000, dep_var='target')
    cat_nn.remove('time_id')
    cat_nn.remove('row_id')
    procs_nn = [Categorify,FillMissing, Normalize]
    dls = TabularPandas(df_train, procs_nn, cat_nn, cont_nn,
                      splits=splits, y_names='target').dataloaders(1024)

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(direction="minimize", pruner=pruner, study_name = 'tabular_learner_50_epochs', storage='sqlite:///optuna.db',load_if_exists=True)

    study.optimize(functools.partial(train_network, dls=dls))

