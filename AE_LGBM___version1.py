#coding=UTF-8

# ====================================================
# Library
# ====================================================
import os, time
import gc
import pathlib
import warnings
warnings.filterwarnings('ignore')
import random
import scipy as sp
import numpy as np
import pandas as pd
import joblib
import itertools
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from itertools import combinations
from glob import glob

from colorama import Fore, Back, Style
c_  = Fore.GREEN
y_  = Fore.YELLOW
b_  = Fore.CYAN
sr_ = Style.RESET_ALL

import logging
logging.basicConfig(level=logging.INFO,
                    filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    #format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
                    format='%(asctime)s - %(levelname)s -:: %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"logger started. LGBM ---> 🔴🟡🟢 ")



# ====================================================
# Configurations
# ====================================================
class CFG:
    input_dir = './data/output/from_raddar/'
    output_dir = './data/output/lgbm'
    seed = 19791895
    n_folds = 10
    target = 'target'
    boosting_type = 'dart'
    metric = 'binary_logloss'

# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ====================================================
# Read data
# ====================================================
def read_data():
    train = pd.read_parquet(CFG.input_dir + 'train_fe.parquet')
    test = pd.read_parquet(CFG.input_dir + 'test_fe.parquet')
    return train, test

# ====================================================
# Amex metric
# ====================================================
def amex_metric(y_true, y_pred):
    labels = np.transpose(np.array([y_true, y_pred]))
    labels = labels[labels[:, 1].argsort()[::-1]]
    weights = np.where(labels[:,0]==0, 20, 1)
    cut_vals = labels[np.cumsum(weights) <= int(0.04 * np.sum(weights))]
    top_four = np.sum(cut_vals[:,0]) / np.sum(labels[:,0])
    gini = [0,0]
    for i in [1,0]:
        labels = np.transpose(np.array([y_true, y_pred]))
        labels = labels[labels[:, i].argsort()[::-1]]
        weight = np.where(labels[:,0]==0, 20, 1)
        weight_random = np.cumsum(weight / np.sum(weight))
        total_pos = np.sum(labels[:, 0] *  weight)
        cum_pos_found = np.cumsum(labels[:, 0] * weight)
        lorentz = cum_pos_found / total_pos
        gini[i] = np.sum((lorentz - weight_random) * weight)
    return 0.5 * (gini[1]/gini[0] + top_four)

# ====================================================
# LGBM amex metric
# ====================================================
def lgb_amex_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'amex_metric', amex_metric(y_true, y_pred), True



# ====================================================
# Best Model Saving
# ====================================================
SavePath = pathlib.Path("./")

class SaveModelCallback:
    def __init__(self,
                 models_folder: pathlib.Path,
                 fold_id: int,
                 min_score_to_save: float,
                 every_k: int,
                 order: int = 0):
        self.min_score_to_save: float = min_score_to_save
        self.every_k: int = every_k
        self.current_score = min_score_to_save
        self.order: int = order
        self.models_folder: pathlib.Path = models_folder
        self.fold_id: int = fold_id
        self.start_time = time.time()

    def __call__(self, env):
        iteration = env.iteration
        score = env.evaluation_result_list[3][2]
        if iteration % self.every_k == 0:
            print(f'iteration {iteration}, score={score:.05f}')
            if score > self.current_score:
                self.current_score = score
                for fname in self.models_folder.glob(f'lgbm_dart_fold{self.fold_id}_seed{CFG.seed}*'):
                    fname.unlink()
                print(f'{c_}High Score: iteration {iteration}, score={score:.05f}{sr_}')
                logger.info(f'High Score: iteration {iteration}, score={score:.05f}')
                joblib.dump(env.model, self.models_folder / f'lgbm_dart_fold{self.fold_id}_seed{CFG.seed}_best_{score:.05f}.pkl')
        if iteration % 1000 == 0:
            epoch_end_time = time.time()
            epoch_time_elapsed = epoch_end_time - self.start_time
            logger.info('Iteration[{}--{}] complete in {:.0f}h {:.0f}m {:.0f}s ⏱\n'.format(iteration-1000, iteration,
                epoch_time_elapsed // 3600, (epoch_time_elapsed % 3600) // 60, (epoch_time_elapsed % 3600) % 60))
            print('Iteration[{}--{}] complete in {}{:.0f}h {:.0f}m {:.0f}s{}⏱\n'.format(iteration-1000, iteration, y_,
                epoch_time_elapsed // 3600, (epoch_time_elapsed % 3600) // 60, (epoch_time_elapsed % 3600) % 60, sr_))
            self.start_time = epoch_end_time


def save_model(models_folder: pathlib.Path, fold_id: int, min_score_to_save: float = 0.78, every_k: int = 50):
    return SaveModelCallback(models_folder=models_folder, fold_id=fold_id, min_score_to_save=min_score_to_save, every_k=every_k)

# ====================================================
# Train & Evaluate
# ====================================================
def train_and_evaluate(train, test):
    # Label encode categorical features
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
        "D_68"
    ]
    cat_features = [f"{cf}_last" for cf in cat_features]
    for cat_col in cat_features:
        encoder = LabelEncoder()
        train[cat_col] = encoder.fit_transform(train[cat_col])
        test[cat_col] = encoder.fit_transform(test[cat_col])
    # Round last float features to 2 decimal place
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    num_cols = [col for col in num_cols if 'last' in col]
    for col in num_cols:
        train[col + '_round2'] = train[col].round(2)
        test[col + '_round2'] = test[col].round(2)
    # Get the difference between last and mean
    num_cols = [col for col in train.columns if 'last' in col]
    num_cols = [col[:-5] for col in num_cols if 'round' not in col]
    for col in num_cols:
        try:
            train[f'{col}_last_mean_diff'] = train[f'{col}_last'] - train[f'{col}_mean']
            test[f'{col}_last_mean_diff'] = test[f'{col}_last'] - test[f'{col}_mean']
        except:
            pass
    # Transform float64 and float32 to float16
    num_cols = list(train.dtypes[(train.dtypes == 'float32') | (train.dtypes == 'float64')].index)
    for col in tqdm(num_cols):
        train[col] = train[col].astype(np.float16)
        test[col] = test[col].astype(np.float16)
    # Get feature list
    features = [col for col in train.columns if col not in ['customer_ID', CFG.target]]
    params = {
        'objective': 'binary',
        'metric': CFG.metric,
        'boosting': CFG.boosting_type,
        'seed': CFG.seed,
        'num_leaves': 100,
        'learning_rate': 0.01,
        'feature_fraction': 0.20,
        'bagging_freq': 10,
        'bagging_fraction': 0.50,
        'lambda_l2': 2,
        'min_data_in_leaf': 40,
        'num_threads': 18,
        'device': 'gpu',
        }
    # Create a numpy array to store test predictions
    test_predictions = np.zeros(len(test))
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train))
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(train, train[CFG.target])):
        print(' ')
        print('-'*50)
        print(f'Training fold {fold} with {len(features)} features...')
        logger.info('-'*50)
        logger.info(f'Training fold {fold} with {len(features)} features...')
        fold_start_time = time.time()
        x_train, x_val = train[features].iloc[trn_ind], train[features].iloc[val_ind]
        y_train, y_val = train[CFG.target].iloc[trn_ind], train[CFG.target].iloc[val_ind]
        lgb_train = lgb.Dataset(x_train, y_train, categorical_feature = cat_features)
        lgb_valid = lgb.Dataset(x_val, y_val, categorical_feature = cat_features)
        model = lgb.train(
            params = params,
            train_set = lgb_train,
            num_boost_round = 10500,
            valid_sets = [lgb_train, lgb_valid],
            #early_stopping_rounds = 1500,
            #verbose_eval = 500,
            feval = lgb_amex_metric,
            callbacks=[save_model(models_folder=SavePath, fold_id=fold, min_score_to_save=0.78, every_k=50)],
            )

        # Save last model
        joblib.dump(model, f'{CFG.output_dir}/lgbm_{CFG.boosting_type}_fold{fold}_seed{CFG.seed}.pkl')

        pkl = glob(f"./lgbm_dart_fold{fold}_seed{CFG.seed}*.pkl")
        best_model = joblib.load(pkl[0])

        # Predict validation
        val_pred = best_model.predict(x_val)
        # Add to out of folds array
        oof_predictions[val_ind] = val_pred
        # Predict the test set
        test_pred = best_model.predict(test[features])
        test_predictions += test_pred / CFG.n_folds
        # Compute fold metric
        score = amex_metric(y_val, val_pred)
        print(f'Our fold {fold} CV score is {score}')

        fold_end_time = time.time()
        fold_time_elapsed = fold_end_time - fold_start_time
        logger.info('Fold {} complete in {:.0f}h {:.0f}m {:.0f}s ⏰\n'.format(fold,
            fold_time_elapsed // 3600, (fold_time_elapsed % 3600) // 60, (fold_time_elapsed % 3600) % 60))
        print('Fold {} complete in {}{:.0f}h {:.0f}m {:.0f}s{} ⏰\n'.format(fold, b_,
            fold_time_elapsed // 3600, (fold_time_elapsed % 3600) // 60, (fold_time_elapsed % 3600) % 60, sr_))
        del x_train, x_val, y_train, y_val, lgb_train, lgb_valid
        gc.collect()
    # Compute out of folds metric
    score = amex_metric(train[CFG.target], oof_predictions)
    print(f'Our out of folds CV score is {score}')
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame({'customer_ID': train['customer_ID'], 'target': train[CFG.target], 'prediction': oof_predictions})
    oof_df.to_csv(f'./oof_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    # Create a dataframe to store test prediction
    test_df = pd.DataFrame({'customer_ID': test['customer_ID'], 'prediction': test_predictions})
    test_df.to_csv(f'./test_lgbm_{CFG.boosting_type}_baseline_{CFG.n_folds}fold_seed{CFG.seed}.csv', index = False)
    
seed_everything(CFG.seed)
train, test = read_data()
train_and_evaluate(train, test)

