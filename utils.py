import os
import pickle

import csv
import numpy as np
import pandas as pd
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
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

    