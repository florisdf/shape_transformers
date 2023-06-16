from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm


EPOCH = 'Training Epoch'
L2 = 'L2'


def get_sweep_results(
    sweep_id, project="shape_transformers", entity="florisdf", lazy=False,
):
    res_path = Path(f'sweep_{sweep_id}.pkl')
    if res_path.exists() and lazy:
        return pd.read_pickle(res_path)

    api = wandb.Api()

    results = []

    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = sweep.runs

    for run in tqdm(runs, leave=False):
        df_history = run.history(keys=[
            "ValLoss/L2",
            "epoch",
        ])
        df_history['val_fold'] = run.config['k_fold_val_fold']
        df_history['run_id'] = run.id

        for k, v in run.config.items():
            df_history[k] = run.config[k]

        results.append(df_history)

    df = pd.concat(results, ignore_index=True)
    df.to_pickle(res_path)

    return df