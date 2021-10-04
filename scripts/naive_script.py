import pandas as pd
import numpy as np
import os
from sklearn.metrics import r2_score
import glob


def realized_volatility_single_stock(file_path, prediction_column_name):
    df_book_data = pd.read_parquet(file_path)
    stock_id = file_path.split('=')[1]
    time_ids, bpr, bsz, apr, asz = (df_book_data[col].values for col in ['time_id', 'bid_price1','bid_size1','ask_price1','ask_size1' ])
    wap = (bpr * asz +apr * bsz) / (asz + bsz)
    log_wap = np.log(wap)
    ids, index = np.unique(time_ids, return_index=True)

    splits = np.split(log_wap, index[1:])
    ret=[]
    for time_id, x in zip(ids.tolist(), splits):
        log_ret = np.diff(x)
        volatility = np.sqrt((log_ret ** 2).sum())
        ret.append((f'{stock_id}-{time_id}', volatility.item()))
    return pd.DataFrame(ret, columns=['row_id', prediction_column_name])


def realized_volatility_all(files_list, prediction_column_name):
    return pd.concat( [realized_volatility_single_stock(file, prediction_column_name) for file in files_list])

list_order_book_file_test = glob.glob('/kaggle/input/optiver-realized-volatility-prediction/book_test.parquet/*')
df_naive_pred_test = realized_volatility_all(list_order_book_file_test,'target')
df_naive_pred_test.to_csv('submission.csv',index = False)
fname = 'naive.feather'
df_naive_pred_test[['row_id', 'target']].reset_index().to_feather(fname)