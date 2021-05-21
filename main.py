import csv
import argparse
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
# import torch
# from LambdaRankNN import LambdaRankNN
# import xgboost as xgb
import lightgbm

from LambdaRank import *
from EDA_plots import *
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
    gc.collect()
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
        # df_train = df_train[:10000]
        # df_test = df_test[:10000]
        if args.plot_EDA:
            print_statistics(df_train, df_test)
            # plot_missing_values(df_train, 'train')
            # plot_missing_values(df_test, 'test')
            # create_price_boxplot(df_train)
            # create_dist_boxplot(df_train)
            # create_numerical_boxplot(df_train, boxplot_cols=[], name='numerical')
            # create_rating_boxplot(df_train, boxplot_cols=[], name='rating')
            # create_date_plot(df_train, df_test)
            # create_boolean_plot(df_train)
            # create_position_bias_plot(df_train)
            exit()
        if 'score' not in df_train.columns:
            add_score_column(df_train)
            pdump(df_train,'df_train')
        print("Preprocessing datasets...")
        preprocess_data(df_train), preprocess_data(df_test)
        if args.add_proba_features:
            df_train, df_test = add_probability_features(df_train, df_test)
            gc.collect()
        if args.add_stats_features:
            add_position_features(df_train, df_test)
            add_statistics_num_features(df_train, df_test)
            gc.collect()
        if args.add_comp_features:
            add_composite_features(df_train, df_test)
            gc.collect()
        
        X_train, y_train, qids_train, X_val, y_val, qids_val, categorical_numbers, col_names = create_train_val_data(df_train)
        gc.collect()
        X_test, qids_test, prop_ids_test, categorical_numbers = create_test_data(df_test)
        gc.collect()

    elif args.load_subsets:
        print("Loading subsets...")
        X_train, y_train, qids_train, X_val, y_val, qids_val = load_subsets()
        categorical_numbers = None
        X_test = pload('X_test')
        qids_test = pload('qids_test')
        prop_ids_test = pload('prop_ids_test')      

    print("#" * 80)
    print("Shape of train set: ", X_train.shape)
    print("Shape of val set: ", X_val.shape)
    print("Shape of test set: ", X_test.shape)

    _, group_train = np.unique(qids_train, return_counts=True)
    _, group_val = np.unique(qids_val, return_counts=True)
    if args.standardize:
        X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)

    if args.model == 'boost_tree':
        print("Initializing tree-based model...")
        model = lightgbm.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=2000,
            learning_rate=0.10,
            # bagging_fraction=0.75,
            max_position=5,
            # label_gain=[0, 1, 5],
            random_state=args.seed,
            boosting_type='dart',
        )
        gc.collect()

        model.fit(
            X_train,
            y_train,
            group=group_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_group=[group_train, group_val],
            eval_names=['Training set', 'Validation set'],
            feature_name=col_names,
            verbose=100,
            eval_at=5,
            early_stopping_rounds=500,
            categorical_feature=categorical_numbers,
        )
        plot_training(model)
        plot_importance(model)

    elif args.model == 'lambdarank':
        print("=============== Start of training LTR model ===============", flush=True)
        model = LambdaRankNN(input_size=X_train.shape[1], 
                        hidden_layer_sizes=(128, 256), 
                        activation=('relu', 'relu'), 
                        solver='adam')

        model.fit(X_train, y_train, qids_train, epochs=5)
        print("Now starting Evaluation", flush=True)
        model.evaluate(X_val, y_val, qids_val, eval_at=5)
        print("#" * 80)

    print("Creating test results...")
    test_scores = model.predict(X_test)

    test_df = pd.DataFrame(columns=['prop_id','srch_id','score'])
    test_df['prop_id'] = prop_ids_test
    test_df['srch_id'] = qids_test
    test_df['score'] = test_scores

    test_df = test_df.sort_values(["srch_id", "score"], ascending=[True, False])
    
    print("Saving predictions into submission.csv")
    if not os.path.exists('./submission/'):
        os.makedirs('./submission/')
    timestr = time.strftime("%Y_%m_%d-%H_%M_%S")
    test_df[["srch_id", "prop_id"]].to_csv(os.path.join("./submission/submission_{}_{}.csv".format(args.model, timestr)), index=False)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducability')
    parser.add_argument('--model', type=str, default='boost_tree', help='Model used for training')
    parser.add_argument('--plot_EDA', action='store_true')
    parser.add_argument('--load_subsets', action='store_true')
    parser.add_argument('--standardize', action='store_true')
    parser.add_argument('--add_stats_features', action='store_true')
    parser.add_argument('--add_proba_features', action='store_true')
    parser.add_argument('--add_comp_features', action='store_true')
    args = parser.parse_args()
    main(args)