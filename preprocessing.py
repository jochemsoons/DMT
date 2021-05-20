import csv
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import gc
from scipy.stats.mstats import winsorize
from sklearn import preprocessing

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
    print("Adding mean, median and std for numeric features...")
    combined_data = pd.concat([df_train, df_test], copy=False)
    numeric_features = ["position", "prop_starrating", "prop_review_score", "prop_location_score1", "prop_location_score2"]
    print("Numeric features used:", numeric_features)
    for feature in numeric_features:
        mean = combined_data.groupby("prop_id")[feature].mean().fillna(value=-1)
        median = combined_data.groupby("prop_id")[feature].median().fillna(value=-1)
        std = combined_data.groupby("prop_id")[feature].std().fillna(value=-1)

        for df in (df_train, df_test):
            df[feature + "_mean"] = mean[df.prop_id].values
            df[feature + "_median"] = median[df.prop_id].values
            df[feature + "_std"] = std[df.prop_id].values
    return df_train, df_test

def add_probability_features(df_train, df_test):
    print("Adding booking and click probability features...")
    df_train['book_probability'] = df_train.groupby("prop_id")['booking_bool'].transform('sum') / df_train.groupby("prop_id")['prop_id'].transform('count')
    df_train['click_probability'] = df_train.groupby("prop_id")['click_bool'].transform('sum') / df_train.groupby("prop_id")['prop_id'].transform('count')
    
    train_book_prob = df_train[["prop_id", "book_probability"]].drop_duplicates()
    train_click_prob = df_train[["prop_id", "click_probability"]].drop_duplicates()
  
    df_test = pd.merge(df_test, train_book_prob, on='prop_id', how='left')
    df_test = pd.merge(df_test, train_click_prob, on='prop_id', how='left')
    
    mean_book_prob = train_book_prob['book_probability'].mean()
    mean_click_prob = train_click_prob['click_probability'].mean()
    df_test.book_probability.fillna(mean_book_prob, inplace = True)
    df_test.click_probability.fillna(mean_click_prob, inplace = True)
    return df_train, df_test
    # test_prop_ids = np.array(df_test['prop_id'])
    # train_prop_ids = np.array(df_train['prop_id'])       
    # common_ids = np.intersect1d(test_prop_ids, train_prop_ids)
    # df_test['book_probability'] = [df_train.loc[(df_train['prop_id']==p_id)].book_probability.iloc[0] if p_id in common_ids else 0 for p_id in test_prop_ids]
    # df_test['click_probability'] = [df_train.loc[(df_train['prop_id']==p_id)].click_probability.iloc[0] if p_id in common_ids else 0 for p_id in test_prop_ids]
    # return df_train, df_test

def add_composite_features(df_train, df_test):
    for df in (df_train, df_test):
        df['starrating_diff'] = abs(df['visitor_hist_starrating'] - df['prop_starrating'])
        df['usd_diff'] = abs(df["visitor_hist_adr_usd"] - df["price_usd"])
        df['star_review_diff'] = abs(df["prop_review_score"] - df["prop_starrating"])
        df['price_order'] = df.groupby("srch_id")["price_usd"].rank("dense")
        hist_price_normal = np.exp(df['prop_log_historical_price'])
        df['price_diff_recent'] = abs(df['price_usd'] - hist_price_normal)
    return df_train, df_test

def add_date_features(df, datetime_key="date_time", features=["month", "hour", "dayofweek"]):
    dates = pd.to_datetime(df[datetime_key])
    for feature in features:
        if feature == "month":
            df["month"] = dates.dt.month
        elif feature == "dayofweek":
            df["dayofweek"] = dates.dt.dayofweek
        elif feature == "hour":
            df["hour"] = dates.dt.hour
    return df

def get_categorical_column(df):
    categorical_features = [
        "dayofweek",
        "month",
        "hour",
        "prop_country_id",
        "site_id",
        "visitor_location_country_id",
    ]
    categorical_features = [c for c in categorical_features if c in df.columns.values]
    categorical_features_numbers = [df.columns.get_loc(x) for x in categorical_features]
    return categorical_features_numbers

def preprocess_data(df):
    add_date_features(df)
    to_drop = [
        'date_time',
        'site_id',
        'visitor_location_country_id',
        # 'visitor_hist_adr_usd',
        'prop_country_id',
        #'prop_id',
        # 'prop_brand_bool',
        #'promotion_flag',
        # 'srch_destination_id',
        #'random_bool',
    ]
    # Remove columns
    df.drop(to_drop, axis=1, inplace=True)
    # Winsorize price values
    # print('max before:', np.max(df['price_usd'] ))
    df['price_usd'] = winsorize(df['price_usd'], (None, 0.04))
    # print('max after:', np.max(df['price_usd'] ))
    # treatment for missing values
    # Replace NULL with -10 in place
    df.orig_destination_distance.fillna(-1, inplace=True)
    df.visitor_hist_starrating.fillna(-1, inplace=True)
    df.visitor_hist_adr_usd.fillna(-1, inplace=True)
    df.prop_review_score.fillna(-1, inplace=True)

    # Replace a value less than the minimum of training + test data
    df.srch_query_affinity_score.fillna(-350, inplace = True)

    df.prop_location_score2.fillna(0, inplace = True)
    # Replace NULL of competitiors with 0 in place
    for i in range(1,9):
        rate = 'comp' + str(i) + '_rate'
        inv = 'comp' + str(i) + '_inv'
        diff = 'comp' + str(i) + '_rate_percent_diff'
        # df[rate].fillna(0, inplace = True)
        # df[inv].fillna(0, inplace = True)
        # df[diff].fillna(0, inplace = True)
        df.drop([rate, inv, diff], axis=1, inplace=True)
    return df

def split_df(df, subset):
    qids = df['srch_id']
    df.sort_values(by=['srch_id'])
    # df.drop(['srch_id'], axis=1, inplace = True)
    if subset == 'train':
        labels = df['score']
        df.drop(['score', 'booking_bool','click_bool', 'position','gross_bookings_usd'], axis = 1, inplace = True)
        df.drop(['prop_id'], axis = 1, inplace = True)
        categorical_numbers = get_categorical_column(df)
        return np.asarray(df, dtype=np.float64), np.asarray(labels, dtype=np.int64), np.asarray(qids, dtype=np.int64), categorical_numbers
    else:
        prop_ids = df['prop_id']
        df.drop(['prop_id'], axis = 1, inplace = True)
        categorical_numbers = get_categorical_column(df)
        return np.asarray(df, dtype=np.float64), np.asarray(qids, dtype=np.int64), np.asarray(prop_ids, dtype=np.int64), categorical_numbers

def create_train_val_data(df, split_ratio=0.80):
    srch_ids = df['srch_id'].unique()
    random.shuffle(srch_ids)
    split_index = round(split_ratio * len(srch_ids))
    train_ids = srch_ids[:split_index]
    val_ids = srch_ids[split_index:]
    train_df = df.loc[df['srch_id'].isin(train_ids)]
    val_df = df.loc[df['srch_id'].isin(val_ids)]

    print("Splitting into X, y and qids...")
    X_train, y_train, qids_train, categorical_numbers = split_df(train_df, subset='train')
    X_val, y_val, qids_val, categorical_numbers = split_df(val_df, subset='train')

    pdump(X_train, 'X_train')
    pdump(X_val, 'X_val')
    pdump(y_train, 'y_train')
    pdump(y_val, 'y_val')
    pdump(qids_train, 'qids_train')
    pdump(qids_val, 'qids_val')

    return X_train, y_train, qids_train, X_val, y_val, qids_val, categorical_numbers

def create_test_data(df):
    X_test, qids_test, prop_ids_test, categorical_numbers = split_df(df, subset='test')
    pdump(X_test, 'X_test')
    pdump(qids_test, 'qids_test')
    pdump(prop_ids_test, 'prop_ids_test')
    return X_test, qids_test, prop_ids_test, categorical_numbers

def standardize_data(X_train, X_val, X_test):
    print("Standardizing data to have zero mean and unit variance...")
    train_len, val_len, test_len = len(X_train), len(X_val), len(X_test)
    X_combined = np.concatenate((X_train, X_val, X_test), axis=0)
    scaler = preprocessing.StandardScaler().fit(X_combined)
    X_scaled = scaler.transform(X_combined)
    X_train = X_scaled[:train_len]
    X_val = X_scaled[train_len : train_len + val_len]
    X_test = X_scaled[train_len + val_len:]
    return X_train, X_val, X_test
