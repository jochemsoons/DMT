import csv
import argparse
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch
from LambdaRankNN import LambdaRankNN
import xgboost as xgb

from preprocessing import *
from utils import *

def print_flags(args):
    """
    Prints all command line arguments.
    """
    print("#" * 80)
    print("ARGUMENTS")
    for arg in vars(args):
        print(arg, ':', getattr(args, arg))
    print("#" * 80)
    return

def main(args):
    # Print all Flags to confirm parameter settings.
    print_flags(args)
    # Set seed for reproducability.
    set_seed(42)
    # Load data.
    try:
        print("Checking if data has been loaded before...")
        df_train = pload('df_train')
        df_test = pload('df_test')
    except:
        df_train, df_test = load_data()
    # Print statistics of train dataset.
    # print_statistics(df_train)

    df_train = df_train[:100000]
    df_test = df_test[:100000]
    if 'score' not in df_train.columns:
        add_score_column(df_train)
        pdump(df_train,'df_train')

    if args.load_subsets:
        print("Loading subsets...")
        X_train, y_train, qids_train, X_val, y_val, qids_val = load_subsets()
        X_test = pload('X_test')
        qids_test = pload('qids_test')
    else:
        print("Preprocessing datasets...")
        preprocess_data(df_train), preprocess_data(df_test)
        if args.add_proba_features:
            add_probability_features(df_train, df_test)
        if args.add_stats_features:
            add_statistics_num_features(df_train, df_test)
        if args.add_comp_features:
            add_composite_features(df_train, df_test)
        X_train, y_train, qids_train, X_val, y_val, qids_val = create_train_val_data(df_train)
        X_test, qids_test = create_test_data(df_test)
    
    print("Shape of train set: ", X_train.shape)
    print("Shape of val set: ", X_val.shape)
    print("Shape of test set: ", X_test.shape)
    
    print(np.isnan(X_train).any())
    print(np.isnan(y_train).any())
    print(np.isnan(qids_train).any())
    print(np.isnan(X_val).any())
    print(np.isnan(X_test).any())
    
    print(np.max(X_train))
    print(np.max(y_train))
    print(np.max(qids_train))
    print(np.min(X_train))
    print(np.min(y_train))
    print(np.min(qids_train))

    # train model
    model = xgb.XGBRanker(  
    tree_method='gpu_hist',
    booster='gbtree',
    objective='rank:pairwise',
    random_state=42, 
    learning_rate=0.1,
    colsample_bytree=0.9, 
    eta=0.05, 
    max_depth=6, 
    n_estimators=110, 
    subsample=0.75 
    )

    # print("Training LambdaRank model...")
    # ranker = LambdaRankNN(input_size=X_train.shape[1], hidden_layer_sizes=(64,128,), activation=('relu', 'relu',), solver='adam')
    # ranker.fit(X_train, y_train, qids_train, epochs=5)
    # y_test = ranker.predict(X_test)
    # print(y_test)
    # print(y_test.shape)
    # print("Evaluating on val set...")
    # ranker.evaluate(X_val, y_val, qids_val, eval_at=5)

    # # generate query data
    # X = np.array([[0.2, 0.3, 0.4],
    #             [0.1, 0.7, 0.4],
    #             [0.3, 0.4, 0.1],
    #             [0.8, 0.4, 0.3],
    #             [0.9, 0.35, 0.25]])
    # y = np.array([0, 1, 0, 0, 2])
    # qid = np.array([1, 1, 1, 2, 2])

    # # train model
    # ranker = LambdaRankNN(input_size=X.shape[1], hidden_layer_sizes=(16,8,), activation=('relu', 'relu',), solver='adam')
    # ranker.fit(X, y, qid, epochs=5)
    # y_pred = ranker.predict(X)
    # ranker.evaluate(X, y, qid, eval_at=2)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_subsets', action='store_true')
    parser.add_argument('--add_stats_features', action='store_true')
    parser.add_argument('--add_proba_features', action='store_true')
    parser.add_argument('--add_comp_features', action='store_true')
    args = parser.parse_args()
    main(args)