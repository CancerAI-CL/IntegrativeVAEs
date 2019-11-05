from misc.helpers import get_data
import os
import glob
import pandas as pd
import numpy as np

type_to_data = {
    'ER': r"../data/5-fold_ERstratified",
    'IC': r"../data/5-fold_ic10stratified",
    'PAM': r"../data/5-fold_pam50stratified",
    'DR': r"../data/5-fold_DRstratified",
    'W': r"../data/",
    
}


class Dataset:
    def __init__(self, dtype, fold):
        self.type = dtype
        self.fold = fold
        self.train, self.test = self._get_data(dtype, fold)

    def _get_data(self, dtype, fold):
        foldpath = os.path.join(type_to_data[dtype], "fold" + fold)
        dev_file = glob.glob(foldpath + "/*test.csv")
        train_file = glob.glob(foldpath + "/*train.csv")

        for file_ in dev_file:
            dev = pd.read_csv(file_, index_col=None, header=0)
        for file_ in train_file:
            train = pd.read_csv(file_, index_col=None, header=0)
        return get_data(train), get_data(dev)

class DatasetWhole:
    def __init__(self, dtype):
        self.type = dtype
        self.train = self._get_data(dtype)

    def _get_data(self, dtype):
        foldpath = os.path.join(type_to_data[dtype])
        train_file = glob.glob(foldpath + "/*.csv")

        for file_ in train_file:
            train = pd.read_csv(file_, index_col=None, header=0)
        return get_data(train)