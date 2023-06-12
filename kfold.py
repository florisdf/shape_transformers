from copy import deepcopy

import numpy as np
import pandas as pd


def kfold_split(ds_train, k=5, val_fold=0, seed=15):
    """
    Shuffle the data in `ds_train`, split it up into into `k` folds and assign
    `k-1` folds to the training dataset and 1 fold to the validation dataset.
    Which fold is used for validation, is determined by `val_fold`. The random
    state used to shuffle the data is set by `seed`.
    """
    assert val_fold < k
    ds_val = deepcopy(ds_train)

    df_train = ds_train.df
    df_val = ds_val.df

    labels = df_train['label'].unique()
    np.random.seed(seed)
    np.random.shuffle(labels)
    label_folds = np.array_split(labels, k)

    val_labels = label_folds.pop(val_fold)
    train_labels = np.concatenate(label_folds)
    df_train = df_train[df_train['label'].isin(train_labels)]
    df_val = df_val[df_val['label'].isin(val_labels)]

    ds_train.df = df_train.reset_index(drop=True)
    ds_val.df = df_val.reset_index(drop=True)

    return ds_train, ds_val
