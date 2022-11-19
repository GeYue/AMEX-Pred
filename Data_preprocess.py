#coding=UTF-8

# ====================================================
# Library
# ====================================================
import gc
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
import itertools

PARQUET_BASE_PATH = f"./data/raddar"
DEFAULT_BASE_PATH = f"./data/default"



DS_CFG = {
    "AfterPay": False,
    "StateDate": True,
}

# ====================================================
# Get the difference
# ====================================================
def get_difference(data, num_features):
    df1 = []
    customer_ids = []
    for customer_id, df in tqdm(data.groupby(['customer_ID'])):
        # Get the differences
        diff_df1 = df[num_features].diff(1).iloc[[-1]].values.astype(np.float32)
        # Append to lists
        df1.append(diff_df1)
        customer_ids.append(customer_id)
    # Concatenate
    df1 = np.concatenate(df1, axis = 0)
    # Transform to dataframe
    df1 = pd.DataFrame(df1, columns = [col + '_diff1' for col in df[num_features].columns])
    # Add customer id
    df1['customer_ID'] = customer_ids
    return df1

# ====================================================
# Read & preprocess data and save it to disk
# ====================================================
def read_preprocess_data():
    train = pd.read_parquet(f'{PARQUET_BASE_PATH}/train.parquet') 

    ################################################
    # Compute train dataset "after pay" features
    ################################################
    if DS_CFG['AfterPay']:
        for bcol in [f'B_{i}' for i in [1,2,3,4,5,9,11,14,17,24]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
            for pcol in ['P_2','P_3']:
                if bcol in train.columns:
                    train[f'{bcol}-{pcol}'] = train[bcol] - train[pcol]

    if DS_CFG['StateDate']:
        import cudf
        temp = train[["customer_ID","S_2"]]
        temp['S_2'] = cudf.to_datetime(temp.S_2).to_numpy()
        temp["SDist"] = temp.groupby("customer_ID")["S_2"].diff() / np.timedelta64(1, 'D')
        temp["SDist"].fillna(30.53, inplace=True)
        train = pd.concat([train, temp["SDist"]], axis=1)

    features = train.drop(['customer_ID', 'S_2'], axis = 1).columns.to_list()

    cat_features = [
        "B_30",
        "B_38",
        "D_114",
        "D_116",
        "D_117",
        "D_120",
        "D_126",
        "D_63",
        "D_64",
        "D_66",
        "D_68",
    ]
    num_features = [col for col in features if col not in cat_features]

    ################
    ### Train FE ###
    ################
    print('Starting train feature extraction')
    train_num_agg = train.groupby("customer_ID")[num_features].agg(['first', 'mean', 'std', 'min', 'max', 'last'])
    train_num_agg.columns = ['_'.join(x) for x in train_num_agg.columns]
    train_num_agg.reset_index(inplace = True)

    # Lag Features
    for col in train_num_agg:

        ## generate "_last_first_sub & _last_first_div"
        if 'last' in col and col.replace('last', 'first') in train_num_agg:
            train_num_agg[col + '_first_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'first')]
            train_num_agg[col + '_first_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'first')]

        ## generate "_last_mean_sub & _last_mean_div"
        if 'last' in col and col.replace('last', 'mean') in train_num_agg:
            train_num_agg[col + '_mean_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'mean')]
            train_num_agg[col + '_mean_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'mean')]

        ## generate "_last_max_sub & _last_max_sub"
        if 'last' in col and col.replace('last', 'max') in train_num_agg:
            train_num_agg[col + '_max_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'max')]
            train_num_agg[col + '_max_sub'] = train_num_agg[col] / train_num_agg[col.replace('last', 'max')]

        ## generate "_last_min_sub & _last_min_div"
        if 'last' in col and col.replace('last', 'min') in train_num_agg:
            train_num_agg[col + '_min_sub'] = train_num_agg[col] - train_num_agg[col.replace('last', 'min')]
            train_num_agg[col + '_min_div'] = train_num_agg[col] / train_num_agg[col.replace('last', 'min')]


    train_cat_agg = train.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
    train_cat_agg.columns = ['_'.join(x) for x in train_cat_agg.columns]
    train_cat_agg.reset_index(inplace = True)
    train_labels = pd.read_csv(f'{DEFAULT_BASE_PATH}/train_labels.csv')

    ##train['dayofweek'] = [pd.Timestamp(cf).dayofweek for cf in train['S_2']]
    ##train['dayofmonth'] = [pd.Timestamp(cf).day for cf in train['S_2']]
    ##train_day_info = train[['customer_ID', 'dayofweek', 'dayofmonth']]

    # Transform float64 columns to float32
    cols = list(train_num_agg.dtypes[train_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        train_num_agg[col] = train_num_agg[col].astype(np.float32)
    # Transform int64 columns to int32
    cols = list(train_cat_agg.dtypes[train_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        train_cat_agg[col] = train_cat_agg[col].astype(np.int32)

    # Get the difference
    train_diff = get_difference(train, num_features)
    train = train_num_agg.merge(train_cat_agg, how = 'inner', on = 'customer_ID').merge(train_diff, how = 'inner', on = 'customer_ID').merge(train_labels, how = 'inner', on = 'customer_ID')
    del train_num_agg, train_cat_agg, train_diff
    gc.collect()
    
    
    ###############
    ### Test FE ###
    ###############
    test = pd.read_parquet(f'{PARQUET_BASE_PATH}/test.parquet')

    ###############################################
    # Compute test dataset "after pay" features
    ###############################################
    if DS_CFG['AfterPay']:
        for bcol in [f'B_{i}' for i in [1,2,3,4,5,9,11,14,17,24]]+['D_39','D_131']+[f'S_{i}' for i in [16,23]]:
            for pcol in ['P_2','P_3']:
                if bcol in test.columns:
                    test[f'{bcol}-{pcol}'] = test[bcol] - test[pcol]

    if DS_CFG['StateDate']:
        temp = test[["customer_ID","S_2"]]
        temp['S_2'] = cudf.to_datetime(temp.S_2).to_numpy()
        temp["SDist"] = temp.groupby("customer_ID")["S_2"].diff() / np.timedelta64(1, 'D')
        temp["SDist"].fillna(30.53, inplace=True)
        test = pd.concat([test, temp["SDist"]], axis=1)

    print('Starting test feature extraction')
    test_num_agg = test.groupby("customer_ID")[num_features].agg(['first', 'mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]
    test_num_agg.reset_index(inplace = True)

    # Lag Features
    for col in test_num_agg:

        if 'last' in col and col.replace('last', 'first') in test_num_agg:
            test_num_agg[col + '_first_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'first')]
            test_num_agg[col + '_first_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'first')]

        if 'last' in col and col.replace('last', 'mean') in test_num_agg:
            test_num_agg[col + '_mean_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'mean')]
            test_num_agg[col + '_mean_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'mean')]

        if 'last' in col and col.replace('last', 'max') in test_num_agg:
            test_num_agg[col + '_max_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'max')]
            test_num_agg[col + '_max_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'max')]

        if 'last' in col and col.replace('last', 'min') in test_num_agg:
            test_num_agg[col + '_min_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'min')]
            test_num_agg[col + '_min_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'min')]

    test_cat_agg = test.groupby("customer_ID")[cat_features].agg(['count', 'first', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]
    test_cat_agg.reset_index(inplace = True)

    ##test['dayofweek'] = [pd.Timestamp(cf).dayofweek for cf in test['S_2']]
    ##test['dayofmonth'] = [pd.Timestamp(cf).day for cf in test['S_2']]
    ##test_day_info = test[['customer_ID', 'dayofweek', 'dayofmonth']]

    # Transform float64 columns to float32
    cols = list(test_num_agg.dtypes[test_num_agg.dtypes == 'float64'].index)
    for col in tqdm(cols):
        test_num_agg[col] = test_num_agg[col].astype(np.float32)
    # Transform int64 columns to int32
    cols = list(test_cat_agg.dtypes[test_cat_agg.dtypes == 'int64'].index)
    for col in tqdm(cols):
        test_cat_agg[col] = test_cat_agg[col].astype(np.int32)

    # Get the difference
    test_diff = get_difference(test, num_features)
    test = test_num_agg.merge(test_cat_agg, how = 'inner', on = 'customer_ID').merge(test_diff, how = 'inner', on = 'customer_ID')
    del test_num_agg, test_cat_agg, test_diff
    gc.collect()


    # Save files to disk
    train.to_parquet(f'./data/output/train_fe.parquet')
    test.to_parquet(f'./data/output/test_fe.parquet')

# Read & Preprocess Data
read_preprocess_data()
