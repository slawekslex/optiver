import pandas as pd
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from fastai.basics import *


def wap1(df):
    return (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (df['bid_size1'] + df['ask_size1'])
def wap2(df):
    return (df['bid_price2'] * df['ask_size2'] + df['ask_price2'] * df['bid_size2']) / (df['bid_size2'] + df['ask_size2'])
def wap_balance(df): return df['wap1'] - df['wap2']
def price_spread(df): return (df['ask_price1'] - df['bid_price1']) / ((df['ask_price1'] + df['bid_price1']) / 2)
def bid_spread(df): return df['bid_price1'] - df['bid_price2']
def ask_spread(df): return df['ask_price1'] - df['ask_price2']
def total_volume(df): return df['ask_size1'] + df['ask_size2'] + df['bid_size1'] + df['bid_size2']
def volume_imbalance(df): return df['ask_size1'] + df['ask_size2'] - df['bid_size1'] - df['bid_size2']   
def log_return(series):
    return np.log(series).diff()
def log_return1(df): return df.groupby(['time_id'])['wap1'].apply(log_return)
def log_return2(df): return df.groupby(['time_id'])['wap2'].apply(log_return)
def log_return_price(df): return df.groupby('time_id')['price'].apply(log_return)
                                         

def realized_volatility(series):
    return np.sqrt(np.sum(series**2))

def process_data(df, feature_dict, windows):
    ret=[]
    string_feature_dict = dict(feature_dict)
    for f in feature_dict.keys():
        if callable(f):
            df[f.__name__] = f(df)
            string_feature_dict[f.__name__] = string_feature_dict.pop(f)
    for w in windows:
        data = df[(df.seconds_in_bucket >= w[0]) & (df.seconds_in_bucket < w[1])]
        w_suff = f'_{w[0]}_{w[1]}'
        time_ids = data.time_id.unique()
        df_feature= data.groupby('time_id').agg(string_feature_dict)
        df_feature.columns = ['_'.join(x) + w_suff for x in df_feature.columns]
        ret.append(df_feature)
    return pd.concat(ret, axis=1)
                

def ffill(data_df):
    data_df=data_df.set_index(['time_id', 'seconds_in_bucket'])
    data_df = data_df.reindex(pd.MultiIndex.from_product([data_df.index.levels[0], np.arange(0,600)], names = ['time_id', 'seconds_in_bucket']), method='ffill')
    return data_df.reset_index()



class OptiverFeatureGenerator():
    def __init__(self, book_feature_dict, trade_feature_dict, time_windows, 
                 time_id_features, time_id_agg, stock_id_features, stock_id_agg,
                 data_dir = Path('../input/optiver-realized-volatility-prediction'),
                 cache_dir = None):
        store_attr()

        
    def process_one_stock(self, stock_id, typ='train'):
        book_df = pd.read_parquet(self.data_dir / f'book_{typ}.parquet/stock_id={stock_id}')
        trade_df = pd.read_parquet(self.data_dir / f'trade_{typ}.parquet/stock_id={stock_id}')
        book_df = ffill(book_df)
        book_feat = process_data(book_df, self.book_feature_dict, self.time_windows)
        trade_feat = process_data(trade_df, self.trade_feature_dict, self.time_windows)
        ret = pd.concat([trade_feat,book_feat], axis=1)
        ret = ret.reset_index()
        ret['row_id'] = ret['time_id'].apply(lambda x:f'{stock_id}-{x}')
        ret = ret.drop('time_id', axis=1)
        return ret

    def process_all(self, list_stock_ids, typ='train'):
        pool = Pool()
        df = pool.starmap(self.process_one_stock, zip(list_stock_ids, [typ]*len(list_stock_ids)))
        df = pd.concat(df, ignore_index = True)
        return df

    def add_time_stock(self, df):
        if self.stock_id_features:
            # Group by the stock id
            df_stock_id = df.groupby(['stock_id'])[self.stock_id_features].agg(self.stock_id_agg).reset_index()
            # Rename columns joining suffix
            df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
            df_stock_id = df_stock_id.add_suffix('_' + 'stock')
            df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
            df.drop(['stock_id__stock'], axis = 1, inplace = True)
        if self.time_id_features:
            df_time_id = df.groupby(['time_id'])[self.time_id_features].agg(self.time_id_agg).reset_index()
            # Rename columns joining suffix
            df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
            df_time_id = df_time_id.add_suffix('_' + 'time')
            df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
            df.drop(['time_id__time'], axis = 1, inplace = True)
        return df

    def generate_test_df(self):
        test = pd.read_csv(self.data_dir/'test.csv')
        test_stock_ids = test['stock_id'].unique()
        test_features = self.process_all(test_stock_ids, 'test')
        test = test.merge(test_features, on = ['row_id'], how = 'left')
        test =  self.add_time_stock(test)
        return test

    def generate_train_df(self):
        train_df = pd.read_csv(self.data_dir/'train.csv')
        train_df['row_id'] = train_df['stock_id'].astype(str) + '-' + train_df['time_id'].astype(str)
        train_stock_ids = train_df['stock_id'].unique()
        train_features = self.process_all(train_stock_ids, 'train')
        train_df = train_df.merge(train_features, on = ['row_id'], how = 'left')
        train_df = self.add_time_stock(train_df)
        return train_df