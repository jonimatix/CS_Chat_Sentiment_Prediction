import pandas as pd
import numpy as np
import datetime
import pickle
import dill


def get_current_datetime_num(format="%Y%m%d_%H_%M_%S"):
    return datetime.datetime.now().strftime(format)


def save_class_obj(path, obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        dill.dump(obj, f)


def load_class_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return dill.load(f)


def save_obj(path, obj, name):
    with open(path + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path, name):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2

    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def chunker(seq, size):
    ''' Splits the data frame into chunks of size N '''
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


'''
Usage: 
df = pd.DataFrame(np.random.rand(14,4), columns=['a', 'b', 'c', 'd'])

for i in chunker(df2, 6): # Chunk of size n
    print(i)
'''


def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "Please input a DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)

    return df[indices_to_keep].astype(np.float64)


'''
np.isnan(X_train).sum().sort_values()
np.where(np.isnan(X_train))
np.nan_to_num(X_train)

np.any(np.isnan(X_train))
np.all(np.isfinite(X_train))
np.where(np.isfinite(X_train))

X_train[np.isfinite(X_train)]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(999, inplace=True)
'''


