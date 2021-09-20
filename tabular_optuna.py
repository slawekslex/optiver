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

activations = {'relu': nn.ReLU(True), 'leaky_relu': nn.LeakyReLU(True), 'silu': nn.SiLU(True)}
optimizers = {'adam':Adam, 'ranger':ranger, 'rmsprop': RMSProp}
schedules = ['one_cycle', 'flat_cos']

def train_network(trial, dls, activation, optimizer, schedule, train_epochs, save_as=None):
    n_layers = trial.suggest_int('n_layers',2,4)
    dropouts = []
    layer_sizes = []
    for i in range(n_layers):
        p = 0 if i==0 else trial.suggest_float(f'p{i}', 0, .5)
        dropouts.append(p)
        layer_sizes.append(trial.suggest_int(f'layer_{i}', 10, 1000))
    stock_embed_size = trial.suggest_int('stock_emb', 3, 30)
    embed_p = trial.suggest_float('embed_p', 0, .5)
    config = {'ps': dropouts, 'embed_p':embed_p, 'lin_first':False,  'act_cls':activation}
    lr = trial.suggest_float('lr', 1e-3, 5e-2)
    learn = tabular_learner(dls, y_range=(0,.1), layers=layer_sizes, emb_szs={'stock_id':stock_embed_size},
                        n_out=1, loss_func = rmspe, metrics=AccumMetric(rmspe), config=config, opt_func = optimizer)
    
    print(learn.model)
    
    # with learn.no_bar():
    #     with learn.no_logging():
    if schedule == 'one_cycle':
        learn.fit_one_cycle(train_epochs, lr)
    if schedule == 'flat_cos':
        learn.fit_flat_cos(train_epochs, lr)
    if save_as:
        learn.save(save_as)
    last3 = L(learn.recorder.values).itemgot(2)[-3:]
    return np.mean(last3)


def train_cross_valid(trial, dlss):
    res = 0
    activation = trial.suggest_categorical('activation', activations.keys())
    optimizer = trial.suggest_categorical('optimizer', optimizers.keys())
    if optimizer == 'ranger':
        schedule = 'flat_cos'
    else:
        schedule = trial.suggest_categorical('schedule', schedules)
    train_epochs = trial.suggest_int('epochs',8, 30)
    bs =trial.suggest_categorical('bs', [256, 512, 1024, 2048])
    print('using ', activation, optimizer, schedule, train_epochs, bs)
    for idx, dls in enumerate(dlss):
        dls.train.bs=bs
        v = train_network(trial, dls, activations[activation], optimizers[optimizer], schedule, train_epochs, save_as = f'optuned_{idx}')
        print(f'fold {idx}: {v}')
        trial.report(v, step = idx)
        if trial.should_prune():
            raise optuna.TrialPruned(f'Trial was pruned after fold {idx}')
        res += v/5

    print('cross valid:', res)
    return res

if __name__ == '__main__':
    df_train = pd.read_csv('train_with_features.csv')
    df_train = df_train.fillna(0)
    print('loading datasets...')
    dlss = [get_dls(df_train, [list(trn_idx), list(val_idx)]) for trn_idx, val_idx in GroupKFold().split(df_train, groups = df_train.time_id)]
    print('creating study..')


    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="minimize", study_name = 'tabular_learner_features', storage='sqlite:///optuna.db',load_if_exists=True, pruner=pruner)

    #study.optimize(functools.partial(train_cross_valid, dlss=dlss))
    best_trial = study.best_trial
    print(best_trial)
    train_cross_valid(best_trial, dlss)

