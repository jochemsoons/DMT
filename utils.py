import os
import pickle

import time
import csv
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import lightgbm
# import torch
import random


def pdump(values, filename, dirname='./pickle_dumps/') :
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    print("Dumping " + filename)
    pickle.dump(values, open(os.path.join(dirname + filename + '_pdump.pkl'), 'wb'))

def pload(filename, dirname='./pickle_dumps/') :
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    file = os.path.join(dirname + filename + '_pdump.pkl')
    if not os.path.isfile(file) :
        raise FileNotFoundError(file + " doesn't exist")
    print("Loading " + filename)
    return pickle.load(open(file, 'rb'))

# Function for setting the seed
def set_seed(seed):
    print("Setting seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available(): # GPU operation have separate seed
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

def load_subsets():
    X_train = pload('X_train')
    X_val = pload('X_val')
    y_train = pload('y_train')
    y_val = pload('y_val')
    qids_train = pload('qids_train')
    qids_val = pload('qids_val')   
    return X_train, y_train, qids_train, X_val, y_val, qids_val


def print_statistics(df_train, df_test):
    cols_unique = [
                    'srch_id',
                    'site_id',
                    'visitor_location_country_id',
                    'prop_country_id',
                    'prop_id',
                    'srch_destination_id',
        ]
    for i, df in enumerate([df_train, df_test]):
        if i == 0:
            print("TRAIN SET STATISTICS:")
            # train_stats = df.describe()
            # print('variables', df.columns)
            print('shape of dataframe', df.shape)
            print('# of logs', len(df))
            print('# of not clicked', len(df.loc[(df['click_bool']==0)]))
            print('# of clicked', len(df.loc[(df['click_bool']==1)]))
            print('# of clicked and not booked', len(df.loc[(df['click_bool']==1) & (df['booking_bool']==0)]))
            print('# of clicked and booked', len(df.loc[(df['click_bool']==1) & (df['booking_bool']==1)]))
        else:
            print("TEST SET STATISTICS:")
            # test_stats = df.describe()
            # print('variables', df.columns)
            print('shape of dataframe', df.shape)
            print('# of logs', len(df))

    for col in cols_unique:
        unique_train = df_train[col].unique()
        unique_test =  df_test[col].unique()
        train_count = len(set(unique_train))
        test_count = len(set(unique_test))
        intersect = list(set(unique_train) & set(unique_test))
        shared_count = len(intersect)
        union = list(set(unique_train) | set(unique_test))
        combined_count = len(union)
        print("Feature: {} \n# unique in train: {} \n# unique in test: {} \ntotal # of unique values: {} \n# of shared values: {}\n".format(col, train_count, test_count, combined_count, shared_count))
    
def plot_training(model):
    if not os.path.exists('./train_plots/'):
        os.makedirs('./train_plots/')
    ax = lightgbm.plot_metric(model)
    ax.set_title("NDGC@5 scores of LGBMRanker during training")
    ax.set_ylabel("NDGC@5 score")
    ax.set_xlabel("Number of estimators (iterations)")
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    plt.savefig('./train_plots/tree_training_{}.png'.format(timestr))

def plot_importance(model):
    if not os.path.exists('./train_plots/'):
        os.makedirs('./train_plots/')
    ax = lightgbm.plot_importance(model, figsize=(12,4), grid=False, max_num_features=15)
    ax.set_title("Attribute importance derived from training", fontsize=20)
    ax.set_ylabel("Attribute", fontsize=16)
    ax.set_xlabel("LGBMRanker importance value", fontsize=16)
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    plt.tight_layout()
    plt.savefig('./train_plots/feature_importance{}.png'.format(timestr))


