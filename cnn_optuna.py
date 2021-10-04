from fastai.vision.all import *
from fastai.tabular.all import *
from sklearn.model_selection import KFold, GroupKFold
import optuna


class FastAIPruningCallback(TrackerCallback):
  
    def __init__(self, trial: optuna.Trial, monitor: str = "valid_loss"):
        super().__init__(monitor=monitor)
        self.report_epoch = 0
        self.trial = trial

    def after_epoch(self) -> None:
        super().after_epoch()
        self.report_epoch +=1
        
        self.trial.report(self.recorder.final_record[self.idx], step=self.report_epoch)
        if self.trial.should_prune():
            raise CancelFitException()

    def after_fit(self) -> None:
        super().after_fit()
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned at epoch {self.report_epoch}.")

PATH = Path('../input/optiver-realized-volatility-prediction')


def add_target_bin(train_df):
    w = train_df.target.to_numpy()

    w =np.sort(w)

    bins = []
    bin_med=[]
    step = (len(w)+9)//10
    for i in range(0, len(w), step):
        j = min(i+step, len(w))
        bins.append(w[j] if j< len(w) else 1)
        bin_med.append(np.median(w[i:j]))
        #print(w[i], w[j-1],np.median(w[i:j]) )

    target_bin = np.digitize(train_df.target, bins)

    train_df['target_bin']=target_bin
    return train_df

def get_dls(train_ftrs, trn_idx, val_idx, target_category):
    if target_category:
        target = 'target_bin'
        train_ftrs = train_ftrs.drop('target', axis=1)
    else:
        target = 'target'
        train_ftrs = train_ftrs.drop('target_bin', axis=1)
    cont_nn,cat_nn = cont_cat_split(train_ftrs, max_card=9000, dep_var=target)
    cont_nn.remove('offset')
    cat_nn=[x for x in cat_nn if not x in ['row_id', 'time_id']]
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(train_ftrs, procs_nn, cat_nn, cont_nn,
                        splits=[list(trn_idx), list(val_idx)], y_names=target)
    return to_nn.dataloaders(1024, after_batch = ReadBatch)

class ReadBatch(ItemTransform):
    def encodes(self, to):
        book_offsets = torch.tensor(to['offset'].to_numpy()).long()
        book_data = torch_data.view(-1,600,8)[book_offsets//600,:,:]
        book_data = book_data.permute(0,2,1)
        res = (tensor(to.cats).long(),tensor(to.conts).float(), book_data)        
        res = res + (tensor(to.targ),)
        if to.device is not None: res = to_device(res, to.device)
        return res

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(ch, ch, kernel_size = 5, padding = 2, padding_mode='replicate'),
            nn.BatchNorm1d(ch),
            nn.ReLU(),
            nn.Conv1d(ch, ch, kernel_size = 5, padding = 2, padding_mode='replicate'),
            nn.BatchNorm1d(ch),
        )
        
    def forward(self, x):
        res = self.layers(x) + x
        res = F.relu(res)
        return res

class ResnetModel(nn.Module):
    def __init__(self, num_outputs, chan=20, conv_depth=6, res_width=1,p=.1, do_sigmoid = False):
        super().__init__()
        self.do_sigmoid = do_sigmoid
        layers = [nn.Conv1d(8, chan, kernel_size=1), nn.BatchNorm1d(chan) ,nn.ReLU()]
        
        for _ in range(conv_depth):
            layers += [ResBlock(chan) for _ in range(res_width)]
            layers += [nn.AvgPool1d(3, padding=1)]
        layers += [Flatten(), nn.Dropout(p)]   
        self.conv_layers = nn.Sequential(*layers)
        test_x = torch.ones(32,8,600)
        conv_out = self.conv_layers(test_x).shape[1]
        self.classifier = nn.Linear(conv_out, num_outputs)
        
    def forward(self, x_cat, x_cont, x_raw):
        feat = self.conv_layers(x_raw)
        res = self.classifier(feat)
        if self.do_sigmoid:
            res = sigmoid_range(res, 0, .1).view(-1)
        return res

class ConvFeatModel(nn.Module):
    def __init__(self, emb_szs, n_cont, layer_sizes, conv_layers, embed_p,ps):
        super().__init__()
        self.embeds = nn.ModuleList([Embedding(ni, nf) for ni,nf in emb_szs])
        self.emb_drop = nn.Dropout(embed_p)
        self.conv_layers = conv_layers
        self.bn_cont = nn.BatchNorm1d(n_cont)
        test_x = torch.ones(32,8,600)
        conv_out = self.conv_layers.cuda()(test_x.cuda()).shape[1]
        n_emb = sum(e.embedding_dim for e in self.embeds)
        sizes = [n_emb + n_cont + conv_out] + layer_sizes + [1]
        actns = [nn.ReLU() for _ in range(len(sizes)-2)] + [None]
        layers = [LinBnDrop(sizes[i], sizes[i+1], bn = (i!=len(actns)-1), p=p, act=a, lin_first=False)
                       for i,(p,a) in enumerate(zip(ps+[0.],actns))]
        layers.append(SigmoidRange(0, 0.1))
        self.layers = nn.Sequential(*layers)
    def forward(self, x_cat, x_cont, x_raw):
        x = [e(x_cat[:,i]) for i,e in enumerate(self.embeds)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x_cont = self.bn_cont(x_cont)
        x_conv = self.conv_layers(x_raw)
        x = torch.cat([x, x_cont, x_conv], 1)
        return self.layers(x)

def CE_loss(inp, tar):
    return F.cross_entropy(inp, tar.view(-1).long())



def split_2way(model):
    #return L(params(model.initial_conv)+params(model.conv_layers), params(model.classifier))
    return L(params(model.conv_layers), params(model.layers)+params(model.embeds))
def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()

def pretrain(trial, dls):
    chan =  trial.suggest_int('chan', 8, 64)
    conv_depth = trial.suggest_int('conv_depth', 4, 7)
    res_width = trial.suggest_int('res_width', 1, 3)
    p= trial.suggest_float('conv_p', 0, .5)
    epochs = 10
 
    model = ResnetModel(10, chan=chan,conv_depth=conv_depth, res_width=res_width, p=p)
    learn = Learner(dls, model, metrics = [accuracy], loss_func = CE_loss)
    learn.fit_one_cycle(epochs, 1e-3)
    result = L(learn.recorder.values).itemgot(2)[-1]
    print('reporting', 1/result)
    trial.report(1/result, step=0)
    if trial.should_prune():
            raise optuna.TrialPruned(f"Trial was pruned after pertrain.")
    return learn.model.conv_layers



def train_fold(trial, dls_cat, dls_reg):
    
    conv_layers = pretrain(trial, dls_cat)
    stock_emb = trial.suggest_int('stock_emb',5, 30)
    emb_sizes = [(len(dls_reg.train.classes['stock_id']), stock_emb)]
    n_cont = len(dls_reg.cont_names)
    max_sizes = [1000,500,200]
    layer_sizes = [trial.suggest_int(f'layer{i}', 10, mx) for i, mx in enumerate(max_sizes)]
    embed_p = trial.suggest_float('embed_p', 0, .5)
    ps = [trial.suggest_float(f'p{i}', 0, .5) for i in range(3)]
    freeze_epochs = 3
    lr_mult = trial.suggest_int('lr_mult', 1, 100)
    lr = trial.suggest_float('lr', 1e-3, 1e-2)
    wd = trial.suggest_float('wd', 0, 0.5)
    model = ConvFeatModel(emb_sizes, n_cont, layer_sizes, conv_layers, embed_p,ps)
    callback =FastAIPruningCallback(trial, monitor="rmspe")
    learn = Learner(dls_reg,model, loss_func=rmspe, splitter = split_2way, metrics=AccumMetric(rmspe),
        cbs=callback)
    
    learn.fine_tune(50, lr, freeze_epochs=freeze_epochs, lr_mult=lr_mult, wd=wd)
    
    last3 = L(learn.recorder.values).itemgot(2)[-3:]
    return np.mean(last3)

if __name__ == '__main__':
    train_ftrs = pd.read_feather('train_24cols.feather')
    
    cols_to_keep = ['stock_id', 'row_id', 'time_id', 'target',
       'log_return_price_std_0_600', 'order_count_sum_0_600',
       'seconds_in_bucket_size_0_600', 'size_sum_0_600',
       'log_return1_std_0_600_min_time', 'log_return1_std_0_600_mean_time',
       'log_return1_std_0_600_min_stock', 'log_return1_std_0_600_mean_stock',
       'log_return_price_std_0_600_mean_time',
       'log_return_price_std_200_600_mean_time',
       'log_return_price_std_400_600_mean_time',
       'log_return_price_std_0_600_min_time',
       'log_return_price_std_200_600_min_time',
       'log_return_price_std_400_600_min_time', 'total_volume_mean_0_600',
       'offset']

    
    torch_data = torch.load(PATH/'torch_data2.pth')
    for c in [2,3,6,7]:
        torch_data[:,c] = (1 /  (1+torch_data[:,c])).sqrt()
    means, stds = torch_data.mean(dim=0), torch_data.std(dim=0)
    torch_data = (torch_data - means) / stds
    offset = offset = list(range(0, len(torch_data), 600))
    train_ftrs['offset']=offset
    train_ftrs = train_ftrs[cols_to_keep]
    train_ftrs = add_target_bin(train_ftrs)
    train_ftrs = train_ftrs.fillna(0)

    trn_idx, val_idx = first(GroupKFold().split(train_ftrs, groups = train_ftrs.time_id))

    dls_cat = get_dls(train_ftrs, trn_idx, val_idx, True)
    dls_reg = get_dls(train_ftrs, trn_idx, val_idx, False)

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(direction="minimize", study_name = 'cnn', storage='sqlite:///optuna.db',load_if_exists=True, pruner=pruner)
    study.optimize(functools.partial(train_fold, dls_cat=dls_cat, dls_reg = dls_reg))
    