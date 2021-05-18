import csv
import argparse
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
# import torch
# from LambdaRankNN import LambdaRankNN
# import xgboost as xgb
import lightgbm


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
    if args.seed:
        set_seed(args.seed)
    # Load data.
    if not args.load_subsets:
        try:
            print("Checking if data has been loaded before...")
            df_train = pload('df_train')
            df_test = pload('df_test')
        except:
            df_train, df_test = load_data()
        # Print statistics of train dataset.
        # df_train = df_train
        # df_test = df_test
        if 'score' not in df_train.columns:
            add_score_column(df_train)
            pdump(df_train,'df_train') 

        print("Preprocessing datasets...")
        preprocess_data(df_train), preprocess_data(df_test)
        if args.add_proba_features:
            df_train, df_test = add_probability_features(df_train, df_test)
        if args.add_stats_features:
            add_statistics_num_features(df_train, df_test)
        if args.add_comp_features:
            add_composite_features(df_train, df_test)
        X_train, y_train, qids_train, X_val, y_val, qids_val = create_train_val_data(df_train)
        X_test, qids_test, prop_ids_test = create_test_data(df_test)

    elif args.load_subsets:
        print("Loading subsets...")
        X_train, y_train, qids_train, X_val, y_val, qids_val = load_subsets()
        X_test = pload('X_test')
        qids_test = pload('qids_test')
        prop_ids_test = pload('prop_ids_test')      

    print("#" * 80)
    print("Shape of train set: ", X_train.shape)
    print("Shape of val set: ", X_val.shape)
    print("Shape of test set: ", X_test.shape)

    _, group_train = np.unique(qids_train, return_counts=True)
    _, group_val = np.unique(qids_val, return_counts=True)
    
    print("Initializing model...")
    model = lightgbm.LGBMRanker(
        boosting_type='gbdt',
        random_state=args.seed
    )
    print("Training model...")
    model.fit(
        X_train, 
        y_train, 
        group=group_train, 
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_group=[group_train, group_val],
        verbose=True
    )
    print("#" * 80)
    print("Creating test results...")
    train_scores = model.predict(X_val)
    test_scores = model.predict(X_test)
    test_df = pd.DataFrame(columns=['prop_id','srch_id','score'])
    test_df['prop_id'] = prop_ids_test
    test_df['srch_id'] = qids_test
    test_df['score'] = test_scores

    test_df = test_df.sort_values(["srch_id", "score"], ascending=[True, False])
    
    print("Saving predictions into submission.csv")
    if not os.path.exists('./submission/'):
        os.makedirs('./submission/')
    test_df[["srch_id", "prop_id"]].to_csv(os.path.join("./submission/submission.csv"), index=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducability')
    parser.add_argument('--load_subsets', action='store_true')
    parser.add_argument('--add_stats_features', action='store_true')
    parser.add_argument('--add_proba_features', action='store_true')
    parser.add_argument('--add_comp_features', action='store_true')
    args = parser.parse_args()
    main(args)