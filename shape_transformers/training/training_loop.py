from pathlib import Path
import sys

import torch
from tqdm import tqdm
import wandb

from .training_steps import TrainingSteps


class TrainingLoop:
    def __init__(
        self,
        training_steps: TrainingSteps,
        optimizer,
        lr_scheduler,
        device,
        num_epochs,
        dl_train,
        dl_val,
        save_unique,
        save_last,
        save_best,
        best_metric,
        is_higher_better,
        ckpts_path,
    ):
        self.training_steps = training_steps
        self.model = self.training_steps.model.to(device)
        self.num_epochs = num_epochs
        self.dl_train = dl_train
        self.dl_val = dl_val

        self.minmax_metrics = {}
        self.ckpt_dir = Path(ckpts_path)

        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch_idx = 0
        self.train_batch_idx = -1
        self.val_batch_idx = -1

        self.save_unique = save_unique
        self.save_last = save_last
        self.save_best = save_best
        self.best_metric = best_metric
        self.is_higher_better = is_higher_better

    def run(self):
        self.minmax_metrics = {}

        # Training loop
        for self.epoch_idx in tqdm(range(self.num_epochs), leave=True):
            # Training epoch
            self.model.train()
            self.training_steps.on_before_training_epoch()
            self.training_epoch()
            log_dict = self.training_steps.on_after_training_epoch()
            log(log_dict, epoch_idx=self.epoch_idx, section='Train')

            # Validation epoch
            self.model.eval()
            self.training_steps.on_before_validation_epoch()
            self.validation_epoch()
            log_dict = self.training_steps.on_after_validation_epoch()
            log(log_dict, epoch_idx=self.epoch_idx, section='Val')

            # Update and log minmax_metrics
            self.update_minmax_metrics(log_dict)
            log(self.minmax_metrics, epoch_idx=self.epoch_idx, section='Val')

            # Create checkpoints
            self.create_checkpoints(log_dict)

    def training_epoch(self):
        for (self.train_batch_idx, batch) in enumerate(
            tqdm(self.dl_train, leave=False), start=self.train_batch_idx + 1
        ):
            batch = tuple(x.to(self.device) for x in batch)
            loss, log_dict = self.training_steps.on_training_step(
                batch,
            )
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.model.zero_grad()
            log(log_dict, epoch_idx=self.epoch_idx,
                batch_idx=self.train_batch_idx,
                section='TrainLoss')
        if torch.isnan(loss):
            sys.exit('Loss is NaN. Exiting...')

    def validation_epoch(self):
        # Validation loop
        for batch_idx, batch in tqdm(
            enumerate(self.dl_val),
            leave=False,
            total=len(self.dl_val),
        ):
            self.val_batch_idx += 1
            batch = tuple(x.to(self.device) for x in batch)
            with torch.no_grad():
                self.training_steps.on_validation_step(batch, batch_idx)

    def update_minmax_metrics(self, val_log_dict):
        for k, v in val_log_dict.items():
            if isinstance(v, wandb.Histogram) or isinstance(v, wandb.Object3D):
                continue

            max_name = f'Max{k}'
            if (
                max_name not in self.minmax_metrics
                or v > self.minmax_metrics[max_name]
            ):
                self.minmax_metrics[max_name] = v

            min_name = f'Min{k}'
            if (
                min_name not in self.minmax_metrics
                or v < self.minmax_metrics[min_name]
            ):
                self.minmax_metrics[min_name] = v

    def create_checkpoints(self, val_log_dict=None):
        file_prefix = f"{wandb.run.id}_" if self.save_unique else ""
        file_suffix = '.pth'

        if not self.ckpt_dir.exists():
            self.ckpt_dir.mkdir(parents=True)

        if (self.save_best and val_log_dict is not None):
            if (
                (self.is_higher_better
                    and val_log_dict[self.best_metric] > self.minmax_metrics[f'Max{self.best_metric}'])
                or
                (not self.is_higher_better
                    and val_log_dict[self.best_metric] < self.minmax_metrics[f'Min{self.best_metric}'])
            ):
                torch.save(
                    self.model.state_dict(),
                    self.ckpt_dir / f'{file_prefix}best{file_suffix}'
                )

        if self.save_last:
            torch.save(
                self.model.state_dict(),
                self.ckpt_dir / f'{file_prefix}last{file_suffix}'
            )


def log(log_dict, epoch_idx, batch_idx=None, section=None):
    def get_key(k):
        if section is None:
            return k
        else:
            return f'{section}/{k}'

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.detach().cpu()
        else:
            return v

    for k, v in log_dict.items():
        wandb_dict = {get_key(k): get_value(v),
                      "epoch": epoch_idx}
        if batch_idx is not None:
            wandb_dict['batch_idx'] = batch_idx
        wandb.log(wandb_dict)
