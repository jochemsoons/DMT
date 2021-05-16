import csv
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import gc
from scipy.stats.mstats import winsorize

from utils import *

def load_data(dirname='data/'):
    print("Loading data from CSV files...")
    df_train = pd.read_csv(dirname + 'training_set_VU_DM.csv')
    df_test = pd.read_csv(dirname + 'test_set_VU_DM.csv')
    pdump(df_train, 'df_train')
    pdump(df_test, 'df_test')
    return df_train, df_test

def print_statistics(df):
    print('variables', df.columns)
    print('shape of dataframe', df.shape)
    print('# of logs', len(df))
    print('# of not clicked', len(df.loc[(df['click_bool']==0)]))
    print('# of clicked', len(df.loc[(df['click_bool']==1)]))
    print('# of clicked and not booked', len(df.loc[(df['click_bool']==1) & (df['booking_bool']==0)]))
    print('# of clicked and booked', len(df.loc[(df['click_bool']==1) & (df['booking_bool']==1)]))

def add_score_column(df):
    print("Adding score (label) column...")
    labels = np.zeros((df.shape[0])) 
    df['score'] = labels
    mask_booking = (df['booking_bool'] == 1)
    df['score'][mask_booking] = 5
    mask_click = (df['booking_bool'] != 1) & (df['click_bool'] == 1)
    df['score'][mask_click] = 1
    return df

def add_statistics_num_features(df_train, df_test):
    combined_data = pd.concat([df_train, df_test], copy=False)
    numeric_features = ["position", "prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2"]
    for feature in numeric_features:
        mean = combined_data.groupby("prop_id")[feature].mean().fillna(value=-1)
        median = combined_data.groupby("prop_id")[feature].median().fillna(value=-1)
        std = combined_data.groupby("prop_id")[feature].std().fillna(value=-1)

        for df in (df_train, df_test):
            df[feature + "_mean"] = mean[df.prop_id].values
            df[feature + "_median"] = median[df.prop_id].values
            df[feature + "_std"] = std[df.prop_id].values
    return df_train, df_test

def add_proba_features(df_train, df_test):
    combined_data = pd.concat([df_train, df_test], copy=False)
    # print(combined_data)
    # rows = combined_data.loc[(combined_data['prop_id']==893)]
    print(rows)
    print(rows['booking_bool'])


    print(combined_data['prop_id'] == 893)
    print(combined_data.groupby("prop_id")['booking_bool'].transform('count'))
    print(combined_data.groupby("prop_id")['prop_id'].transform('count'))
    df_train['book_probability'] = df_train.groupby("prop_id")['booking_bool'].transform('count') / df_train.groupby("prop_id")['prop_id'].transform('count')
    print(df_train)
    print(len(booked_count))


def add_features(df):
    df['starrating_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])
    # Add more (composite) features
    # df['occurences'] = 
    return df


def preprocess_data(df):
    # Remove date_time
    df.drop('date_time', axis = 1, inplace = True)
    # Winsorize price values
    # print('max before:', np.max(df['price_usd'] ))
    df['price_usd'] = winsorize(df['price_usd'], (None, 0.03))
    # print('max after:', np.max(df['price_usd'] ))
    # treatment for missing values
    # Replace NULL with -10 in place
    df.orig_destination_distance.fillna(-10,inplace = True)
    df.visitor_hist_starrating.fillna(-10,inplace = True)
    df.visitor_hist_adr_usd.fillna(-10,inplace = True)
    df.prop_review_score.fillna(-10, inplace = True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-350, inplace = True)

    df.prop_location_score2.fillna(0, inplace = True)
    # Replace NULL of competitiors with 0 in place
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        df[rate].fillna(0, inplace = True)
        df[inv].fillna(0, inplace = True)
        df[diff].fillna(0, inplace = True)
    return df

def split_df(df, subset):
    qids = df['srch_id']
    df.drop(['srch_id'], axis=1, inplace = True)
    if subset == 'train':
        labels = df['score']
        df.drop(['score', 'booking_bool','click_bool', 'position','gross_bookings_usd'], axis = 1, inplace = True)
        df.drop(['prop_id'], axis = 1, inplace = True)
        return np.asarray(df), labels, qids
    else:
        df.drop(['prop_id'], axis = 1, inplace = True)
        return np.asarray(df), qids

def create_train_val_data(df, feature_engineering, split_ratio=0.80):
    if feature_engineering:
        add_features(df)
    srch_ids = df['srch_id'].unique()
    random.shuffle(srch_ids)
    split_index = round(split_ratio * len(srch_ids))
    train_ids = srch_ids[:split_index]
    val_ids = srch_ids[split_index:]
    train_df = df.loc[df['srch_id'].isin(train_ids)]
    val_df = df.loc[df['srch_id'].isin(val_ids)]

    print("Splitting into X, y and qids...")
    X_train, y_train, qids_train = split_df(train_df, subset='train')
    X_val, y_val, qids_val = split_df(val_df, subset='train')

    pdump(X_train, 'X_train')
    pdump(X_val, 'X_val')
    pdump(y_train, 'y_train')
    pdump(y_val, 'y_val')
    pdump(qids_train, 'qids_train')
    pdump(qids_val, 'qids_val')

    return X_train, y_train, qids_train, X_val, y_val, qids_val

def create_test_data(df, feature_engineering):
    if feature_engineering:
        add_features(df)
    X_test, qids_test = split_df(df, subset='test')
    pdump(X_test, 'X_test')
    pdump(qids_test, 'qids_test')
    return X_test, qids_test