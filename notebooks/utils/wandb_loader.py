from pathlib import Path

import wandb
import pandas as pd
from tqdm import tqdm


EPOCH = 'Training Epoch'
L2 = 'L2'


def get_run_results(
    run_id, project="shape_transformers", entity="florisdf", lazy=False,
):
    res_path = Path(f'run_{run_id}.pkl')
    if res_path.exists() and lazy:
        return pd.read_pickle(res_path)

    api = wandb.Api()

    run = api.run(f"{entity}/{project}/{run_id}")

    df = run.history(keys=[
        "Val/L2",
        "epoch",
    ])
    df['val_fold'] = run.config['k_fold_val_fold']
    df['run_id'] = run.id

    for k, v in run.config.items():
        df[k] = run.config[k]

    df.to_pickle(res_path)

    return df
