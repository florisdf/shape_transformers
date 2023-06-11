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

    folds = np.array_split(
        ds_train.df.sample(frac=1.0, random_state=seed),
        k
    )
    df_val = folds.pop(val_fold)
    df_train = pd.concat(folds)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    ds_train.df = df_train
    ds_val.df = df_val
    return ds_train, ds_val