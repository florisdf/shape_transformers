import argparse
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
import wandb

from training import TrainingSteps, TrainingLoop
from kfold import kfold_split
from nphm_dataset import NPHMDataset
from shape_transformer import ShapeTransformer


def run_training(
    # Model
    token_size=64,
    disentangle_style=False,

    # Dataset
    data_path='/apollo/datasets/NPHM',
    scan_type='registration',
    drop_bad_scans=True,
    n_verts_subsample=None,
    subsample_seed=15,

    # Ckpt
    load_ckpt=None,
    save_unique=False,
    save_last=True,
    save_best=True,
    best_metric=None,
    is_higher_better=True,
    ckpts_path='./ckpts',

    # K-Fold
    k_fold_seed=15,
    k_fold_num_folds=5,
    k_fold_val_fold=0,

    # Dataloader
    batch_size=32,
    val_batch_size=32,
    num_workers=8,

    # Optimizer
    lr=0.01,
    momentum=0.95,
    weight_decay=1e-5,

    # Train
    num_epochs=30,

    # Device
    device='cuda',
):
    ds_train = NPHMDataset(
        data_path=Path(data_path),
        subset='train',
        scan_type=scan_type,
        drop_bad=drop_bad_scans,
        n_subsample=n_verts_subsample,
        subsample_seed=subsample_seed
    )
    ds_test = NPHMDataset(
        data_path=Path(data_path),
        subset='test',
        scan_type=scan_type,
        drop_bad=drop_bad_scans,
    )
    ds_train, ds_val = kfold_split(
        ds_train,
        k=k_fold_num_folds,
        val_fold=k_fold_val_fold,
        seed=k_fold_seed,
    )
    ds_val.refresh_vert_idxs()

    ds_val_gal, ds_val_quer = split_gallery_query(ds_val)
    dl_train = DataLoader(
        ds_train,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    dl_val_gal = DataLoader(
        ds_val_gal,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    dl_val_quer = DataLoader(
        ds_val_quer,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=val_batch_size,
        num_workers=num_workers,
    )

    device = torch.device(device)

    model = ShapeTransformer(
        token_size=token_size,
        disentangle_style=disentangle_style,
        num_id_classes=len(ds_train.subject_to_label)
    )

    if load_ckpt is not None:
        model.load_state_dict(torch.load(load_ckpt))

    training_steps = TrainingSteps(model=model)

    optimizer = SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )

    training_loop = TrainingLoop(
        training_steps=training_steps,
        optimizer=optimizer,
        device=device,
        num_epochs=num_epochs,
        dl_train=dl_train,
        dl_val_gal=dl_val_gal,
        dl_val_quer=dl_val_quer,
        save_unique=save_unique,
        save_last=save_last,
        save_best=save_best,
        best_metric=best_metric,
        is_higher_better=is_higher_better,
        ckpts_path=ckpts_path,
    )
    training_loop.run()


def split_gallery_query(ds_val):
    is_neutral = ds_val.df['is_neutral_closed'] | ds_val.df['is_neutral_open']
    df_gal = ds_val.df[is_neutral].reset_index(drop=True)
    df_quer = ds_val.df[~is_neutral].reset_index(drop=True)

    ds_val_gal = ds_val
    ds_val_quer = deepcopy(ds_val)

    ds_val_gal.df = df_gal
    ds_val_quer.df = df_quer

    return ds_val_gal, ds_val_quer




def int_list_arg_type(arg):
    return [int(s) for s in arg.split(',') if len(s.strip()) > 0]


def str_list_arg_type(arg):
    return [s.strip() for s in arg.split(',') if len(s.strip()) > 0]


def crop_box_size_type(arg):
    try:
        value = int(arg)
        return (value, value)
    except ValueError:
        return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model
    parser.add_argument(
        '--token_size', default=64,
        help="Size of tokens used as input into the shape transformer."
    )
    parser.add_argument(
        '--disentangle_style', action='store_true',
        help='If set, disentangle style code into an expression and identity part.'
    )

    # Ckpt
    parser.add_argument(
        '--load_ckpt', default=None,
        help='The path to load model checkpoint weights from.'
    )
    parser.add_argument(
        '--save_unique', action='store_true',
        help=(
            'If set, the created checkpoint(s) will get a unique name '
            'containing its WandB run ID.'
        )
    )
    parser.add_argument(
        '--save_best', action='store_true',
        help='If set, save a checkpoint containg the weights with the best '
        'performance, as defined by --best_metric and --higher_is_better.'
    )
    parser.add_argument(
        '--save_last', action='store_true',
        help='If set, save a checkpoint containing the weights of the last '
        'epoch.'
    )
    parser.add_argument(
        '--best_metric', default='MSE',
        help='If this metric improves, create a checkpoint (when --save_best is set).'
    )
    parser.add_argument(
        '--higher_is_better', action='store_true',
        help='If set, the metric set with --best_metric is better when it inreases.'
    )
    parser.add_argument(
        '--ckpts_path', default='./ckpts',
        help='The directory to save checkpoints.'
    )

    # K-Fold args
    parser.add_argument(
        '--k_fold_seed', default=15,
        help='Seed for the dataset shuffle used to create the K folds.',
        type=int
    )
    parser.add_argument(
        '--k_fold_num_folds', default=5,
        help='The number of folds to use.',
        type=int
    )
    parser.add_argument(
        '--k_fold_val_fold', default=0,
        help='The index of the validation fold. '
        'Should be a value between 0 and k - 1.',
        type=int
    )

    # Dataset
    parser.add_argument(
        '--data_path', default='/apollo/datasets/NPHM',
        help='Path to the NPHM dataset.',
    )
    parser.add_argument(
        '--scan_type', default='registration',
        help='Scan type to use for the input data.',
    )
    parser.add_argument(
        '--keep_bad_scans', action='store_true',
        help='If set, leave bad scans in the dataset.',
    )
    parser.add_argument(
        '--n_verts_subsample', default=None,
        help='Number of vertices to subsample.',
        type=int,
    )
    parser.add_argument(
        '--subsample_seed', default=15,
        help='Random seed to use for shuffling the subsample indices during training.',
        type=int
    )

    # Dataloader args
    parser.add_argument('--batch_size', default=32, help='The training batch size.', type=int)
    parser.add_argument('--val_batch_size', default=32,
                        help='The validation batch size.', type=int)
    parser.add_argument(
        '--num_workers', default=8,
        help='The number of workers to use for data loading.',
        type=int
    )

    # Optimizer args
    parser.add_argument('--lr', default=0.01, help='The learning rate.',
                        type=float)
    parser.add_argument('--momentum', default=0.95, help='The momentum.',
                        type=float)
    parser.add_argument('--weight_decay', default=1e-5, help='The weight decay.',
                        type=float)

    # Train args
    parser.add_argument(
        '--num_epochs', default=500,
        help='The number of epochs to train.',
        type=int
    )

    # Log args
    parser.add_argument(
        '--wandb_entity', help='Weights and Biases entity.'
    )
    parser.add_argument(
        '--wandb_project', help='Weights and Biases project.'
    )
    
    # Device arg
    parser.add_argument('--device', default='cuda',
                        help='The device (cuda/cpu) to use.')

    args = parser.parse_args()

    wandb.init(entity=args.wandb_entity, project=args.wandb_project,
               config=vars(args))
    run_training(
        # Model
        token_size=args.token_size,
        disentangle_style=args.disentangle_style,
    
        # Dataset
        data_path=args.data_path,
        scan_type=args.scan_type,
        drop_bad_scans=not args.keep_bad_scans,
        n_verts_subsample=args.n_verts_subsample,
        subsample_seed=args.subsample_seed,
    
        # Ckpt
        load_ckpt=args.load_ckpt,
        save_unique=args.save_unique,
        save_last=args.save_last,
        save_best=args.save_best,
        best_metric=args.best_metric,
        is_higher_better=args.higher_is_better,
        ckpts_path=args.ckpts_path,
    
        # K-Fold
        k_fold_seed=args.k_fold_seed,
        k_fold_num_folds=args.k_fold_num_folds,
        k_fold_val_fold=args.k_fold_val_fold,
    
        # Dataloader
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_workers=args.num_workers,
    
        # Optimizer
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    
        # Train
        num_epochs=args.num_epochs,
    
        # Device
        device=args.device,
    )
