import csv
import argparse
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import torch

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
        X_train, y_train, qids_train, X_val, y_val, qids_val = load_subsets()
        X_test = pload('X_test')
        qids_test = pload('qids_test')
    else:
        preprocess_data(df_train), preprocess_data(df_test)
        # add_proba_features(df_train, df_test)
        # add_statistics_num_features(df_train, df_test)
        X_train, y_train, qids_train, X_val, y_val, qids_val = create_train_val_data(df_train, feature_engineering=args.feature_engineering)
        X_test, qids_test = create_test_data(df_test, feature_engineering=args.feature_engineering)
    
    print("Shape of train set: ", X_train.shape, y_train.shape, qids_train.shape)
    print("Shape of val set: ", X_val.shape, y_val.shape, qids_val.shape)
    print("Shape of test set: ", X_test.shape, qids_test.shape)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_subsets', action='store_true')
    parser.add_argument('--feature_engineering', action='store_true')
    args = parser.parse_args()

    main(args)