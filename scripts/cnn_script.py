# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:07.875999Z","iopub.execute_input":"2021-09-27T15:43:07.8769Z","iopub.status.idle":"2021-09-27T15:43:08.816235Z","shell.execute_reply.started":"2021-09-27T15:43:07.876817Z","shell.execute_reply":"2021-09-27T15:43:08.815303Z"}}
from fastai.vision.all import *
from fastai.tabular.all import *
from tqdm.notebook import  tqdm
from sklearn.model_selection import GroupKFold
from optiver_features_private import *
PATH = Path('../input/optiver-realized-volatility-prediction')

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.823327Z","iopub.execute_input":"2021-09-27T15:43:08.823725Z","iopub.status.idle":"2021-09-27T15:43:08.845072Z","shell.execute_reply.started":"2021-09-27T15:43:08.823679Z","shell.execute_reply":"2021-09-27T15:43:08.844184Z"}}
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

# %% [code]



# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.847199Z","iopub.execute_input":"2021-09-27T15:43:08.84777Z","iopub.status.idle":"2021-09-27T15:43:08.856882Z","shell.execute_reply.started":"2021-09-27T15:43:08.847705Z","shell.execute_reply":"2021-09-27T15:43:08.855848Z"}}
def get_train_features():
    train_ftrs = pd.read_feather('../input/optiver-private-data/train_24cols.feather').fillna(0)
    train_ftrs['offset']=-1
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

    return train_ftrs[cols_to_keep]

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.858283Z","iopub.execute_input":"2021-09-27T15:43:08.858692Z","iopub.status.idle":"2021-09-27T15:43:08.870865Z","shell.execute_reply.started":"2021-09-27T15:43:08.858654Z","shell.execute_reply":"2021-09-27T15:43:08.869833Z"}}
def get_test_features():
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
    time_windows = [(0,600), (200, 600), (400,600)]
    agg_cols = ['log_return_price_std', 'log_return1_std', 'log_return2_std']
    time_id_features = [f'{col}_{x}_{y}' for x,y in time_windows for col in agg_cols] 
    time_id_aggregations = ['mean', 'std', 'min' ]
    stock_id_features = time_id_features
    stock_id_aggregations = time_id_aggregations
    ofg = OptiverFeatureGenerator(book_feature_dict, trade_feature_dict, time_windows, time_id_features,time_id_aggregations, stock_id_features, stock_id_aggregations)
    test_ftrs =ofg.generate_test_df().fillna(0)

    test_ftrs['offset']=-1
    test_ftrs['target'] = 0.0
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
    
    return test_ftrs[cols_to_keep]

# %% [markdown]
# ### Stats from the train data used for normalization:

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.872326Z","iopub.execute_input":"2021-09-27T15:43:08.872873Z","iopub.status.idle":"2021-09-27T15:43:08.888359Z","shell.execute_reply.started":"2021-09-27T15:43:08.872655Z","shell.execute_reply":"2021-09-27T15:43:08.887649Z"}}
means = torch.load('../input/conv-models/conv_means.pth')
stds = torch.load('../input/conv-models/conv_stds.pth')

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.890986Z","iopub.execute_input":"2021-09-27T15:43:08.891267Z","iopub.status.idle":"2021-09-27T15:43:08.896829Z","shell.execute_reply.started":"2021-09-27T15:43:08.89124Z","shell.execute_reply":"2021-09-27T15:43:08.895918Z"}}
def fix_offsets(data_df):
    offsets = data_df.groupby(['time_id']).agg({'seconds_in_bucket':'min'})
    offsets.columns = ['offset']
    data_df = data_df.join(offsets, on='time_id')
    data_df.seconds_in_bucket = data_df.seconds_in_bucket - data_df.offset
    return data_df

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.898194Z","iopub.execute_input":"2021-09-27T15:43:08.898856Z","iopub.status.idle":"2021-09-27T15:43:08.907436Z","shell.execute_reply.started":"2021-09-27T15:43:08.898815Z","shell.execute_reply":"2021-09-27T15:43:08.906533Z"}}
def ffill(data_df):
    data_df=data_df.set_index(['time_id', 'seconds_in_bucket'])
    data_df = data_df.reindex(pd.MultiIndex.from_product([data_df.index.levels[0], np.arange(0,600)], names = ['time_id', 'seconds_in_bucket']), method='ffill')
    return data_df.reset_index()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.911125Z","iopub.execute_input":"2021-09-27T15:43:08.911537Z","iopub.status.idle":"2021-09-27T15:43:08.919079Z","shell.execute_reply.started":"2021-09-27T15:43:08.911505Z","shell.execute_reply":"2021-09-27T15:43:08.918032Z"}}
def load_data(fname):
    data = pd.read_parquet(fname)
    stock_id = str(fname).split('=')[1]
    time_ids = data.time_id.unique()
    row_ids = list(map(lambda x:f'{stock_id}-{x}', time_ids))
    data = fix_offsets(data)
    data = ffill(data)
    data = data[['bid_price1', 'ask_price1', 'bid_size1', 'ask_size1','bid_price2', 'ask_price2', 'bid_size2', 'ask_size2']].to_numpy()
    data = torch.tensor(data.astype('float32'))
    for c in [2,3,6,7]:
        data[:,c] = (1 /  (1+data[:,c])).sqrt()
    data = (data - means) / stds
    return data, row_ids

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.921018Z","iopub.execute_input":"2021-09-27T15:43:08.921378Z","iopub.status.idle":"2021-09-27T15:43:08.931635Z","shell.execute_reply.started":"2021-09-27T15:43:08.921341Z","shell.execute_reply":"2021-09-27T15:43:08.930494Z"}}
class ReadBatch(ItemTransform):
    def encodes(self, to):
        book_offsets = torch.tensor(to['offset'].to_numpy()).long()
        book_data = torch_data.view(-1,600,8)[book_offsets//600,:,:]
        book_data = book_data.permute(0,2,1)
        res = (tensor(to.cats).long(),tensor(to.conts).float(), book_data)        
        res = res + (tensor(to.targ),)
        if to.device is not None: res = to_device(res, to.device)
        return res
    
def get_dls(train_ftrs, trn_idx, val_idx):
    target = 'target' 
    cont_nn,cat_nn = cont_cat_split(train_ftrs, max_card=9000, dep_var=target)
    if 'offset' in cont_nn: cont_nn.remove('offset')
    cat_nn=[x for x in cat_nn if not x in ['row_id', 'time_id', 'offset']]
    procs_nn = [Categorify, Normalize]
    to_nn = TabularPandas(train_ftrs, procs_nn, cat_nn, cont_nn,
                        splits=[list(trn_idx), list(val_idx)], y_names=target)
    return to_nn.dataloaders(1024, after_batch = ReadBatch)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:08.933113Z","iopub.execute_input":"2021-09-27T15:43:08.933645Z","iopub.status.idle":"2021-09-27T15:43:12.731599Z","shell.execute_reply.started":"2021-09-27T15:43:08.933607Z","shell.execute_reply":"2021-09-27T15:43:12.730694Z"}}
train_ftrs = get_train_features()
trn_idx, val_idx = first(GroupKFold().split(train_ftrs, groups = train_ftrs.time_id))
torch_data = torch.zeros(600,8)
dls = get_dls(train_ftrs, trn_idx, val_idx)

# %% [code]


# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:12.732883Z","iopub.execute_input":"2021-09-27T15:43:12.733232Z","iopub.status.idle":"2021-09-27T15:43:12.739262Z","shell.execute_reply.started":"2021-09-27T15:43:12.733193Z","shell.execute_reply":"2021-09-27T15:43:12.738467Z"}}
def rmspe(preds, targs):
    x = (targs-preds)/targs
    return (x**2).mean().sqrt()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:12.740618Z","iopub.execute_input":"2021-09-27T15:43:12.741359Z","iopub.status.idle":"2021-09-27T15:43:12.757127Z","shell.execute_reply.started":"2021-09-27T15:43:12.741318Z","shell.execute_reply":"2021-09-27T15:43:12.755292Z"}}
def get_preds(df_ftrs, data_dir, model):
    global torch_data
    all_preds = []
    for stock_id in df_ftrs.stock_id.unique():
        data, row_ids = load_data(data_dir/f'stock_id={stock_id}')
        torch_data = data
        ftrs_stock = df_ftrs[df_ftrs.stock_id==stock_id].copy()
        ftrs_stock['offset'] = list(range(0, 600*len(ftrs_stock), 600))
        if len(df_ftrs)==3: torch_data = torch.cat([torch_data]*3)
        test_dl = dls.test_dl(ftrs_stock)
        preds,targs=[],[]
        for batch in test_dl:
            bx1, bx2, bx3, by = [x.cuda() for x in batch]
            with torch.no_grad():
                pred = model(bx1,bx2,bx3)
            preds.append(pred)
            targs.append(by)
        all_preds +=preds
    return all_preds

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:12.75847Z","iopub.execute_input":"2021-09-27T15:43:12.75905Z","iopub.status.idle":"2021-09-27T15:43:12.772586Z","shell.execute_reply.started":"2021-09-27T15:43:12.75901Z","shell.execute_reply":"2021-09-27T15:43:12.771437Z"}}
test_data_dir = PATH/'book_test.parquet'
train_data_dir = PATH/'book_train.parquet'

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:43:12.774073Z","iopub.execute_input":"2021-09-27T15:43:12.774627Z","iopub.status.idle":"2021-09-27T15:43:13.223141Z","shell.execute_reply.started":"2021-09-27T15:43:12.774589Z","shell.execute_reply":"2021-09-27T15:43:13.221624Z"}}
test_ftrs = get_test_features()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:46:08.704389Z","iopub.execute_input":"2021-09-27T15:46:08.704732Z","iopub.status.idle":"2021-09-27T15:46:09.596089Z","shell.execute_reply.started":"2021-09-27T15:46:08.704698Z","shell.execute_reply":"2021-09-27T15:46:09.595247Z"}}
res = []
for i in range(5):
    model_file = f'../input/conv-models/conv_feat_{i}.pth'
    model = torch.load(model_file)
    model=model.eval().cuda()
    all_preds = get_preds(test_ftrs, test_data_dir, model)
    all_preds = torch.cat(all_preds).view(-1)
    res.append(all_preds)

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:46:10.901433Z","iopub.execute_input":"2021-09-27T15:46:10.901795Z","iopub.status.idle":"2021-09-27T15:46:10.91527Z","shell.execute_reply.started":"2021-09-27T15:46:10.901738Z","shell.execute_reply":"2021-09-27T15:46:10.914441Z"}}
res = torch.stack(res).mean(dim=0)


# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:46:13.301461Z","iopub.execute_input":"2021-09-27T15:46:13.301811Z","iopub.status.idle":"2021-09-27T15:46:13.307512Z","shell.execute_reply.started":"2021-09-27T15:46:13.301776Z","shell.execute_reply":"2021-09-27T15:46:13.306369Z"}}
test_ftrs['target']=res.cpu().numpy()

# %% [code] {"execution":{"iopub.status.busy":"2021-09-27T15:46:13.68753Z","iopub.execute_input":"2021-09-27T15:46:13.687883Z","iopub.status.idle":"2021-09-27T15:46:13.697762Z","shell.execute_reply.started":"2021-09-27T15:46:13.687849Z","shell.execute_reply":"2021-09-27T15:46:13.696702Z"}}


fname = 'cnn.feather'
test_ftrs[['row_id', 'target']].reset_index().to_feather(fname)

# %% [code]
