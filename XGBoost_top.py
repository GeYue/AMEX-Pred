#coding=UTF-8

# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
import cupy, cudf # GPU libraries
import matplotlib.pyplot as plt, gc, os
import glob
import xgboost as xgb

print('RAPIDS version',cudf.__version__)

# VERSION NAME FOR SAVED MODEL FILES
VER = 10
# TRAIN RANDOM SEED
SEED = 20220807
# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8

def read_file(path = '', usecols = None):
    # LOAD DATAFRAME
    if usecols is not None: df = cudf.read_parquet(path, columns=usecols)
    else: df = cudf.read_parquet(path)
    # REDUCE DTYPE FOR CUSTOMER AND DATE
    df['customer_ID'] = df['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
    df.S_2 = cudf.to_datetime( df.S_2 )
    # SORT BY CUSTOMER AND DATE (so agg('last') works correctly)
    #df = df.sort_values(['customer_ID','S_2'])
    #df = df.reset_index(drop=True)
    # FILL NAN
    df = df.fillna(NAN_VALUE) 
    print('shape of data:', df.shape)
    
    return df

def process_and_feature_engineer(df):
    # FEATURE ENGINEERING FROM 
    # https://www.kaggle.com/code/huseyincot/amex-agg-data-how-it-created

    # 0. Generate StatementDate info.
    temp = df[["customer_ID","S_2"]]
    temp['S_2'] = cudf.to_datetime(temp.S_2).to_numpy()
    temp["SDist"] = temp.groupby("customer_ID")["S_2"].diff() / np.timedelta64(1, 'D')
    temp["SDist"].fillna(30.53, inplace=True)
    df = cudf.concat([df, temp['SDist']], axis=1)

    all_cols = [c for c in list(df.columns) if c not in ['customer_ID','S_2']]
    cat_features = ["B_30","B_38","D_114","D_116","D_117","D_120","D_126","D_63","D_64","D_66","D_68"]
    num_features = [col for col in all_cols if col not in cat_features]

    test_num_agg = df.groupby("customer_ID")[num_features].agg(['first', 'mean', 'std', 'min', 'max', 'last'])
    test_num_agg.columns = ['_'.join(x) for x in test_num_agg.columns]

    test_cat_agg = df.groupby("customer_ID")[cat_features].agg(['count', 'last', 'nunique'])
    test_cat_agg.columns = ['_'.join(x) for x in test_cat_agg.columns]

    ## Adding new features which come from kaggle other notebooks
    # 1. Lag Features1
    for col in test_num_agg:
        if 'last' in col and col.replace('last', 'first') in test_num_agg:
            test_num_agg[col + '_first_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'first')]
            #test_num_agg[col + '_lag_div'] = test_num_agg[col] / test_num_agg[col.replace('last', 'first')]

        if 'last' in col and col.replace('last', 'mean') in test_num_agg:
            test_num_agg[col + '_mean_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'mean')]

        if 'last' in col and col.replace('last', 'max') in test_num_agg:
            test_num_agg[col + '_max_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'max')]

        if 'last' in col and col.replace('last', 'min') in test_num_agg:
            test_num_agg[col + '_min_sub'] = test_num_agg[col] - test_num_agg[col.replace('last', 'min')]

    # 3. After Pay
    # diff_cols_a = [f"B_{i}" for i in [11, 14, 17]] + ["D_39", "D_131"] + [f"S_{i}" for i in [16, 23]]
    # diff_cols_b = ["P_2", "P_3"]
    # diff_feat = df[['customer_ID'] + diff_cols_a + diff_cols_b]
    # for a in diff_cols_a:
    #     for b in diff_cols_b:
    #             diff_feat[f"{a}-{b}"] = diff_feat[a] - diff_feat[b]
    # diff_feat.drop(diff_cols_a + diff_cols_b, axis=1, inplace=True)
    # diff_feat = diff_feat.groupby('customer_ID').agg(['first', 'mean', 'std', 'min', 'max', 'last'])
    # diff_feat.columns = ['_'.join(x) for x in diff_feat.columns]
    # test_num_agg = cudf.concat([test_num_agg, diff_feat], axis=1)

    # 2. Get the difference between last and mean
    # num_cols = [col for col in test_num_agg.columns if 'last' in col]
    # num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    # for col in num_cols:
    #     try:
    #         test_num_agg[f'{col}_last_mean_diff'] = test_num_agg[f'{col}_last'] - test_num_agg[f'{col}_mean']
    #     except:
    #         pass

    df = cudf.concat([test_num_agg, test_cat_agg], axis=1)
    del test_num_agg, test_cat_agg
    print('shape after engineering', df.shape)
    
    return df


# ====================================================
# Process and Feature Engineer Test Data
# ====================================================
# CALCULATE SIZE OF EACH SEPARATE TEST PART
def get_rows(customers, test, NUM_PARTS = 4, verbose = ''):
    chunk = len(customers)//NUM_PARTS
    if verbose != '':
        print(f'We will process {verbose} data as {NUM_PARTS} separate parts.')
        print(f'There will be {chunk} customers in each part (except the last part).')
        print('Below are number of rows in each part:')
    rows = []

    for k in range(NUM_PARTS):
        if k==NUM_PARTS-1: cc = customers[k*chunk:]
        else: cc = customers[k*chunk:(k+1)*chunk]
        s = test.loc[test.customer_ID.isin(cc)].shape[0]
        rows.append(s)
    if verbose != '': print( rows )
    return rows,chunk

TRAIN_PATH = './data/raddar/train.parquet'
train = read_file(path = TRAIN_PATH)

train = process_and_feature_engineer(train)

# ADD TARGETS
targets = cudf.read_csv('./data/default/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

# COMPUTE SIZE OF 4 PARTS FOR TEST DATA
NUM_PARTS = 4
TEST_PATH = './data/raddar/test.parquet'


# NEEDED TO MAKE CV DETERMINISTIC (cudf merge above randomly shuffles rows)
train = train.sort_index().reset_index()

# FEATURES
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')

del train
_ = gc.collect()

print(f'Reading test data...')
test = read_file(path = TEST_PATH, usecols = ['customer_ID','S_2'])
customers = test[['customer_ID']].drop_duplicates().sort_index().values.flatten()
rows,num_cust = get_rows(customers, test[['customer_ID']], NUM_PARTS = NUM_PARTS, verbose = 'test')


# ====================================================
# Inference test dataset
# ====================================================
# INFER TEST DATA IN PARTS
skip_rows = 0
skip_cust = 0
test_preds = []

paths = glob.glob("./xgboost_dart_10folds/*.xgb")
FOLDS = len(paths)

for k in range(NUM_PARTS):
    
    # READ PART OF TEST DATA
    print(f'\nReading test data...')
    test = read_file(path = TEST_PATH)
    test = test.iloc[skip_rows:skip_rows+rows[k]]
    skip_rows += rows[k]
    print(f'=> Test part {k+1} has shape', test.shape )
    
    # PROCESS AND FEATURE ENGINEER PART OF TEST DATA
    test = process_and_feature_engineer(test)
    if k==NUM_PARTS-1: test = test.loc[customers[skip_cust:]]
    else: test = test.loc[customers[skip_cust:skip_cust+num_cust]]
    skip_cust += num_cust
    
    # TEST DATA FOR XGB
    X_test = test[FEATURES]
    dtest = xgb.DMatrix(data=X_test)
    test = test[['P_2_mean']] # reduce memory
    del X_test
    gc.collect()

    # INFER XGB MODELS ON TEST DATA
    model = xgb.Booster()
    num = 0
    for path in paths:
        model.load_model(path)
        model.set_param({"predictor": "gpu_predictor"})
        if num == 0:
            preds = model.predict(dtest)
            num = 1
        else:
            preds += model.predict(dtest)
    preds /= FOLDS
    test_preds.append(preds)

    # CLEAN MEMORY
    del dtest, model
    _ = gc.collect()


# WRITE SUBMISSION FILE
test_preds = np.concatenate(test_preds)
test = cudf.DataFrame(index=customers,data={'prediction':test_preds})
sub = cudf.read_csv('./data/default/sample_submission.csv')[['customer_ID']]
sub['customer_ID_hash'] = sub['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
sub = sub.set_index('customer_ID_hash')
sub = sub.merge(test[['prediction']], left_index=True, right_index=True, how='left')
sub = sub.reset_index(drop=True)

# DISPLAY PREDICTIONS
sub.to_csv(f'submission_xgb_v{VER}.csv',index=False)
print('Submission file shape is', sub.shape )
print(sub.head())

# # PLOT PREDICTIONS
# plt.hist(sub.to_pandas().prediction, bins=100)
# plt.title('Test Predictions')
# plt.show()
