import csv
import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import random
import calendar


def plot_missing_values(df, name, dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    perc_missing = []
    for var in df.columns:
        perc_missing.append(df[var].isna().sum() / len(df[var]))
        
    labels = list(df.columns)
    labels = [label for _, label in sorted(zip(perc_missing, labels))]
    perc_missing.sort()
    values = perc_missing
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(20,8))
    ax.bar(x, values)
    ax.set_title('Percentage of missing values', fontsize=20)
    ax.set_ylabel('Percentage of values missing', fontsize=16)
    ax.set_xlabel('Attributes', fontsize=16)
    plt.xticks(rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.tight_layout()
    plt.savefig(dirname + 'missing_values_{}.png'.format(name))

def create_price_boxplot(df, name='price', dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    val_col = df['price_usd'].dropna(axis=0)

    plt.figure()
    plt.boxplot(val_col)
    # ax.set_xticklabels(values_dict.keys())
    plt.title('Boxplot of price values')

    plt.ylabel('Price in USD ($)')
    plt.xlabel('Attribute price_usd')
    plt.tight_layout()
    plt.savefig(dirname + 'boxplot_{}.png'.format(name))

def create_dist_boxplot(df, name='distance', dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    val_col = df['orig_destination_distance'].dropna(axis=0)

    plt.figure()
    plt.boxplot(val_col)
    # ax.set_xticklabels(values_dict.keys())
    plt.title('Boxplot of hotel-customer distance values')

    plt.ylabel('Distance between hotel and customer')
    plt.xlabel('Attribute orig_destination_distance')
    plt.tight_layout()
    plt.savefig(dirname + 'boxplot_{}.png'.format(name))

def create_numerical_boxplot(df, boxplot_cols, name='numerical', dirname='./EDA_plots/'):
    boxplot_cols = [
                    # 'visitor_hist_starrating', 
                    'visitor_hist_adr_usd', 
                    # 'prop_starrating', 
                    # 'prop_review_score',
                    'prop_location_score1', 
                    'prop_location_score2', 
                    'prop_log_historical_price', 
                    'srch_length_of_stay', 
                    'srch_booking_window', 
                    'srch_adults_count', 
                    'srch_children_count', 
                    'srch_room_count', 
                    'srch_query_affinity_score', 
                    # 'orig_destination_distance'
                    ]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if 'date_time' in boxplot_cols:
        boxplot_cols.remove('date_time')
    values_dict = {}
    for col in boxplot_cols:
        val_col = df[col].dropna(axis=0)
        values_dict[col] = val_col

    fig, ax = plt.subplots()
    ax.boxplot(values_dict.values())
    ax.set_xticklabels(values_dict.keys())
    ax.set_title('Boxplot of numerical attribute values')
    ax.set_ylabel('Value')
    ax.set_xlabel('Attributes')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dirname + 'boxplot_{}.png'.format(name))

def create_rating_boxplot(df, boxplot_cols, name='rating', dirname='./EDA_plots/'):
    boxplot_cols = [
                    'visitor_hist_starrating', 
                    # 'visitor_hist_adr_usd', 
                    'prop_starrating', 
                    'prop_review_score',
                    # 'prop_location_score1', 
                    # 'prop_location_score2', 
                    # 'prop_log_historical_price', 
                    # 'srch_length_of_stay', 
                    # 'srch_booking_window', 
                    # 'srch_adults_count', 
                    # 'srch_children_count', 
                    # 'srch_room_count', 
                    # 'srch_query_affinity_score', 
                    # 'orig_destination_distance'
                    ]
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if 'date_time' in boxplot_cols:
        boxplot_cols.remove('date_time')
    values_dict = {}
    for col in boxplot_cols:
        val_col = df[col].dropna(axis=0)
        values_dict[col] = val_col

    fig, ax = plt.subplots()
    ax.boxplot(values_dict.values())
    ax.set_xticklabels(values_dict.keys())
    ax.set_title('Boxplot of rating attribute values')
    ax.set_ylabel('Value')
    ax.set_xlabel('Attributes')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(dirname + 'boxplot_{}.png'.format(name))

def create_position_bias_plot(df_original, dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
 
    for random_bool in [0, 1]:
        df = df_original.loc[df_original['random_bool'] == random_bool]
        occurences = df['position'].value_counts().sort_index()
        
        position_book = df.groupby("position")["booking_bool"].sum() / occurences
        position_click = df.groupby("position")["click_bool"].sum() / occurences
        positions = [i for i in range(1,41)]
        x = np.arange(len(positions))  # the label locations
        width = 0.35 # the width of the bars

        fig, ax = plt.subplots(figsize=(12,8))
        rects1 = ax.bar(x - width/2, position_click, width, label='Clicked')
        rects2 = ax.bar(x + width/2, position_book, width, label='Booked')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        if random_bool == 0:
            ax.set_title('Percentage of clicks and bookings versus position (normal ordering)', fontsize=18)
        elif random_bool == 1:
            ax.set_title('Percentage of clicks and bookings versus position (random ordering)', fontsize=18)
        ax.set_ylabel('Percentage of items being clicks and booked', fontsize=14)
        ax.set_xlabel('Position', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(positions)
        ax.legend()

        fig.tight_layout()
        plt.savefig(dirname + 'position_bias_random={}.png'.format(random_bool))  
    
def create_date_plot(df_train, df_test, dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for i, df in enumerate([df_train, df_test]):
        df_time = pd.DataFrame(columns=['srch_id','year','month', 'count'])
        df_time['srch_id'] = df['srch_id']
        df_time["year"] = pd.DatetimeIndex(df['date_time']).year
        df_time["month"] = pd.DatetimeIndex(df['date_time']).month

        df_time = df_time.drop_duplicates(subset='srch_id')
        df_time = df_time.groupby(["year", "month"]).size().reset_index(name='count')
        df_time['month'] = df_time['month'].apply(lambda x: calendar.month_abbr[x])
        df_time["date"] = df_time["month"].astype(str) + ' ' + df_time["year"].astype(str)
        df_time.drop(['year', 'month'], axis=1, inplace=True)
        if i == 0:
            train_dates = df_time['date']
            train_counts = df_time['count']
        elif i == 1:
            test_dates = df_time['date']
            test_counts = df_time['count']

    dates = train_dates
    x = np.arange(len(dates))  # the label locations
    width = 0.35 # the width of the bars

    fig, ax = plt.subplots(figsize=(14,6))
    rects1 = ax.bar(x - width/2, train_counts, width, label='Training set')
    rects2 = ax.bar(x + width/2, test_counts, width, label='Test set')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Number of searches per month', fontsize=20)
    ax.set_ylabel('Total number of searches', fontsize=16)
    ax.set_xlabel('Year and month', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dates)
    ax.legend()

    fig.tight_layout()
    plt.savefig(dirname + 'dates_counts.png')


def create_boolean_plot(df, dirname='./EDA_plots/'):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    boolean_cols = [
                    'prop_brand_bool', 
                    'promotion_flag', 
                    'srch_saturday_night_bool',
                    'random_bool'
                    ]
    pos_bool_ratios = []
    neg_bool_ratios = []
    for col in boolean_cols:
        pos_bool_ratio = df[col].sum() / len(df[col])
        pos_bool_ratios.append(pos_bool_ratio)
        neg_bool_ratios.append(1-pos_bool_ratio)

    x = np.arange(len(boolean_cols))  # the label locations
    width = 0.35 # the width of the bars

    fig, ax = plt.subplots(figsize=(8,4))

    rects1 = ax.bar(x - width/2, neg_bool_ratios, width, label='Bool=0')
    rects2 = ax.bar(x + width/2, pos_bool_ratios, width, label='Bool=1')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_title('Percentage of boolean attributes being True or False')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('Boolean attributes')
    ax.set_xticks(x)
    ax.set_xticklabels(boolean_cols)
    ax.legend()

    fig.tight_layout()
    plt.savefig(dirname + 'boolean_percentages.png')    
    
    

        