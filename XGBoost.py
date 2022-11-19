#coding=UTF-8

# LOAD LIBRARIES
import pandas as pd, numpy as np # CPU libraries
import cupy, cudf # GPU libraries
import matplotlib.pyplot as plt, gc, os

print('RAPIDS version',cudf.__version__)

# VERSION NAME FOR SAVED MODEL FILES
VER = 3
# TRAIN RANDOM SEED
SEED = 20770810
# FILL NAN VALUE
NAN_VALUE = -127 # will fit in int8
# FOLDS PER MODEL
FOLDS = 10

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

print('Reading train data...')
TRAIN_PATH = './data/raddar/train.parquet'
train = read_file(path = TRAIN_PATH)
#train.head()

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

train = process_and_feature_engineer(train)

# ADD TARGETS
targets = cudf.read_csv('./data/default/train_labels.csv')
targets['customer_ID'] = targets['customer_ID'].str[-16:].str.hex_to_int().astype('int64')
targets = targets.set_index('customer_ID')
train = train.merge(targets, left_index=True, right_index=True, how='left')
train.target = train.target.astype('int8')
del targets

# NEEDED TO MAKE CV DETERMINISTIC (cudf merge above randomly shuffles rows)
train = train.sort_index().reset_index()

# FEATURES
FEATURES = train.columns[1:-1]
print(f'There are {len(FEATURES)} features!')

#################################
# HyperOpt Setting
#################################
import time

import sklearn
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error
import xgboost as xgb

XGBOOST_LEARN = False #True
if XGBOOST_LEARN:
    from hyperopt import hp
    from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials
    #define parameter range
    learning_rate=np.linspace(0.01,0.1,10)
    max_depth=np.arange(2, 18, 2)
    colsample_bylevel=np.arange(0.3, 0.8, 0.1)
    iterations=np.arange(50, 1000, 50)
    l2_leaf_reg=np.arange(0,10)
    bagging_temperature=np.arange(0,100,10)
    n_estimators=np.arange(50,500,50)

    # XGB parameters
    xgb_cat_params = {
        'learning_rate':    hp.choice('learning_rate',    learning_rate),
        'max_depth':        hp.choice('max_depth',         max_depth),
        'colsample_bytree': hp.choice('colsample_bytree', colsample_bylevel),
        'n_estimators':     hp.choice('n_estimators',    n_estimators),
        #'loss_function':    'logloss',
        #'nan_mode':         'Min',
        #'task_type':        'GPU',
        'use_label_encoder':False, 
    }

    xgb_fit_params = {
        'eval_metric': 'logloss',
        'early_stopping_rounds': 10,
        'verbose': True #False
    }

    xgb_para = dict()
    xgb_para['cls_params'] = xgb_cat_params
    xgb_para['fit_params'] = xgb_fit_params
    xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

    class HPOpt(object):
        def __init__(self, x_train, x_test, y_train, y_test):
            self.x_train = x_train
            self.x_test  = x_test
            self.y_train = y_train
            self.y_test  = y_test

        def process(self, fn_name, space, trials, algo, max_evals):
            fn = getattr(self, fn_name)
            try:
                print('entering fmin')
                result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
            except Exception as e:
                return {'status': STATUS_FAIL,
                        'exception': str(e)}
            return result

        def xgb_cls(self, para):
            cls = xgb.XGBClassifier(**para['cls_params'])
            print('ctb initialized')
            return self.train_cls(cls, para)

        def train_cls(self, cls, para):
            print('fitting model')
            cls.fit(self.x_train, self.y_train,
                    eval_set=[(self.x_test, self.y_test)],
                    **para['fit_params'])
            print('model fitted')
            pred = cls.predict(self.x_test)
            #loss = para['loss_func'](self.y_test, pred)
            f1=sklearn.metrics.f1_score(self.y_test.to_numpy(),pred)
            f1=f1*(-1)
            print(f1)
            return {'loss': f1, 'status': STATUS_OK}



    X = train.drop(['customer_ID','target'], axis=1)
    Y = train['target']
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)
    obj = HPOpt(X_train, X_test, y_train, y_test)
    xgb_opt = obj.process(fn_name='xgb_cls', space=xgb_para, trials=Trials(), algo=tpe.suggest, max_evals=10)
    print (xgb_opt)

    #Save best parameters in a dictionary
    best_param_xgb={}
    best_param_xgb['learning_rate']=learning_rate[xgb_opt['learning_rate']]
    best_param_xgb['colsample_bytree']=colsample_bylevel[xgb_opt['colsample_bytree']]
    best_param_xgb['max_depth']=max_depth[xgb_opt['max_depth']]
    best_param_xgb['n_estimators']=n_estimators[xgb_opt['n_estimators']]

    print (best_param_xgb)
    exit(-10)

# LOAD XGB LIBRARY
from sklearn.model_selection import KFold
print('XGB Version',xgb.__version__)

# XGB MODEL PARAMETERS
xgb_parms = { 
    'booster': 'dart',
    'rate_drop': 0.1,
    'skip_drop': 0.5,

    'max_depth': 4, 
    'learning_rate': 0.05, 
    'subsample':0.8,
    'colsample_bytree': 0.6, 
    'eval_metric':'logloss',
    'objective':'binary:logistic',
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'random_state':SEED
}

# xgb_parms = {
#     'objective': 'binary:logitraw', 
#     'tree_method': 'gpu_hist',
#     'predictor':'gpu_predictor',
#     'max_depth': 7,
#     'subsample':0.88,
#     'colsample_bytree': 0.1,
#     'gamma':1.5,
#     'min_child_weight':8,
#     'lambda': 50,
#     'eta':0.03,
#     'learning_rate':0.02,
#     'random_state':SEED
# }


# NEEDED WITH DeviceQuantileDMatrix BELOW
class IterLoadForDMatrix(xgb.core.DataIter):
    def __init__(self, df=None, features=None, target=None, batch_size=256*1024):
        self.features = features
        self.target = target
        self.df = df
        self.it = 0 # set iterator to 0
        self.batch_size = batch_size
        self.batches = int( np.ceil( len(df) / self.batch_size ) )
        super().__init__()

    def reset(self):
        '''Reset the iterator'''
        self.it = 0

    def next(self, input_data):
        '''Yield next batch of data.'''
        if self.it == self.batches:
            return 0 # Return 0 when there's no more batch.
        
        a = self.it * self.batch_size
        b = min( (self.it + 1) * self.batch_size, len(self.df) )
        dt = cudf.DataFrame(self.df.iloc[a:b])
        input_data(data=dt[self.features], label=dt[self.target]) #, weight=dt['weight'])
        self.it += 1
        return 1

# https://www.kaggle.com/kyakovlev
# https://www.kaggle.com/competitions/amex-default-prediction/discussion/327534
def amex_metric_mod(y_true, y_pred):

    labels     = np.transpose(np.array([y_true, y_pred]))
    labels     = labels[labels[:, 1].argsort()[::-1]]
    weights    = np.where(labels[:,0]==0, 20, 1)
    cut_vals   = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four   = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])

    gini = [0,0]
    for i in [1,0]:
        labels         = np.transpose(np.array([y_true, y_pred]))
        labels         = labels[labels[:, i].argsort()[::-1]]
        weight         = np.where(labels[:,0]==0, 20, 1)
        weight_random  = np.cumsum(weight / np.sum(weight))
        total_pos      = np.sum(labels[:, 0] *  weight)
        cum_pos_found  = np.cumsum(labels[:, 0] * weight)
        lorentz        = cum_pos_found / total_pos
        gini[i]        = np.sum((lorentz - weight_random) * weight)

    return 0.5 * (gini[1]/gini[0] + top_four)

from colorama import Fore, Back, Style
c_  = Fore.GREEN
y_  = Fore.YELLOW
b_  = Fore.CYAN
sr_ = Style.RESET_ALL

class Plotting(xgb.callback.TrainingCallback):
    
    def __init__(self, rounds, internal):
        self.rounds = rounds
        self.internal = internal

    def after_iteration(self, model, epoch, evals_log):
        if epoch == 0:
            self.grd_start_time = self.start_time = time.time()
            return False
        elif epoch == self.rounds-1:
            end_time = time.time()
            time_elapsed = end_time - self.grd_start_time
            self.grd_start_time = end_time
            print('Total [0 - {}] complete in \t\t{} {:.0f}h {:.0f}m {:.0f}s {} ⏰🪔🕯'.format(self.rounds, y_, 
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60, sr_))
        elif epoch % self.internal == 0:
            end_time = time.time()
            time_elapsed = end_time - self.start_time
            tt_time_elapsed = end_time - self.grd_start_time
            self.start_time = end_time
            print('Epoch [{} - {}] complete in \t\t{} {:.0f}h {:.0f}m {:.0f}s {} \tTotal:: {} {:.0f}h {:.0f}m {:.0f}s {} ⏱'.format(epoch-self.internal, epoch, b_, 
                time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60, sr_,
                c_, tt_time_elapsed // 3600, (tt_time_elapsed % 3600) // 60, (tt_time_elapsed % 3600) % 60, sr_))

        return False

importances = []
oof = []
train = train.to_pandas() # free GPU memory
TRAIN_SUBSAMPLE = 1.0
gc.collect()

skf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
for fold,(train_idx, valid_idx) in enumerate(skf.split(
            train, train.target )):
    
    # TRAIN WITH SUBSAMPLE OF TRAIN FOLD DATA
    if TRAIN_SUBSAMPLE<1.0:
        np.random.seed(SEED)
        train_idx = np.random.choice(train_idx, 
                       int(len(train_idx)*TRAIN_SUBSAMPLE), replace=False)
        np.random.seed(None)
    
    print('#'*50)
    print('### Fold',fold+1)
    print('### Train size',len(train_idx),'Valid size',len(valid_idx))
    print(f'### Training with {int(TRAIN_SUBSAMPLE*100)}% fold data...')
    print('#'*50)
    
    # TRAIN, VALID, TEST FOR FOLD K
    Xy_train = IterLoadForDMatrix(train.loc[train_idx], FEATURES, 'target')
    X_valid = train.loc[valid_idx, FEATURES]
    y_valid = train.loc[valid_idx, 'target']
    
    dtrain = xgb.DeviceQuantileDMatrix(Xy_train, max_bin=256)
    dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
    
    num_boost_round = 1800 #9999
    plotting = Plotting(num_boost_round, internal=100)

    # TRAIN MODEL FOLD K
    model = xgb.train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=num_boost_round,
                early_stopping_rounds=500, #100,
                verbose_eval=100,
                callbacks=[plotting]) 
    model.save_model(f'XGB_v{VER}_fold{fold}.xgb')
    
    # GET FEATURE IMPORTANCE FOR FOLD K
    dd = model.get_score(importance_type='weight')
    df = pd.DataFrame({'feature':dd.keys(),f'importance_{fold}':dd.values()})
    importances.append(df)
            
    # INFER OOF FOLD K
    oof_preds = model.predict(dvalid)
    acc = amex_metric_mod(y_valid.values, oof_preds)
    print('Kaggle Metric =',acc,'\n')
    
    # SAVE OOF
    df = train.loc[valid_idx, ['customer_ID','target'] ].copy()
    df['oof_pred'] = oof_preds
    oof.append( df )
    
    del dtrain, Xy_train, dd, df
    del X_valid, y_valid, dvalid, model
    _ = gc.collect()
    
print('#'*25)
oof = pd.concat(oof,axis=0,ignore_index=True).set_index('customer_ID')
acc = amex_metric_mod(oof.target.values, oof.oof_pred.values)
print('OVERALL CV Kaggle Metric =',acc)

# CLEAN RAM
del train
_ = gc.collect()

oof_xgb = pd.read_parquet(TRAIN_PATH, columns=['customer_ID']).drop_duplicates()
oof_xgb['customer_ID_hash'] = oof_xgb['customer_ID'].apply(lambda x: int(x[-16:],16) ).astype('int64')
oof_xgb = oof_xgb.set_index('customer_ID_hash')
oof_xgb = oof_xgb.merge(oof, left_index=True, right_index=True)
oof_xgb = oof_xgb.sort_index().reset_index(drop=True)
oof_xgb.to_csv(f'oof_xgb_v{VER}.csv',index=False)
oof_xgb.head()

# # PLOT OOF PREDICTIONS
# plt.hist(oof_xgb.oof_pred.values, bins=100)
# plt.title('OOF Predictions')
# plt.show()

# CLEAR VRAM, RAM FOR INFERENCE BELOW
del oof_xgb, oof
_ = gc.collect()


# ====================================================
# Feature Importance
# ====================================================
import matplotlib.pyplot as plt

df = importances[0].copy()
for k in range(1,FOLDS): df = df.merge(importances[k], on='feature', how='left')
df['importance'] = df.iloc[:,1:].mean(axis=1)
df = df.sort_values('importance',ascending=False)
df.to_csv(f'xgb_feature_importance_v{VER}.csv',index=False)

# NUM_FEATURES = 20
# plt.figure(figsize=(10,5*NUM_FEATURES//10))
# plt.barh(np.arange(NUM_FEATURES,0,-1), df.importance.values[:NUM_FEATURES])
# plt.yticks(np.arange(NUM_FEATURES,0,-1), df.feature.values[:NUM_FEATURES])
# plt.title(f'XGB Feature Importance - Top {NUM_FEATURES}')
# plt.show()


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

# COMPUTE SIZE OF 4 PARTS FOR TEST DATA
NUM_PARTS = 4
TEST_PATH = './data/raddar/test.parquet'

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
    model.load_model(f'XGB_v{VER}_fold0.xgb')
    model.set_param({"predictor": "gpu_predictor"})
    preds = model.predict(dtest)
    for f in range(1,FOLDS):
        model.load_model(f'XGB_v{VER}_fold{f}.xgb')
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
