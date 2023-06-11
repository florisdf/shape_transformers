from copy import deepcopy

import torch
import torch.nn.functional as F
import wandb


class TrainingSteps:
    def __init__(
        self,
        model,
    ):
        self.model = model

        self.val_losses = []

    def on_before_training_epoch(self):
        pass

    def on_training_step(self, batch):
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = model(positions, verts)
        else:
            pred_verts = model(positions, verts)

        loss = F.mse_loss(pred_verts, verts)
        log_dict = {
            'MSE': loss,
        }

        return loss, log_dict

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        pass

    def on_validation_step(self, batch):
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = model(positions, verts)
        else:
            pred_verts = model(positions, verts)

        loss = F.mse_loss(pred_verts, verts)
        self.val_losses.append(loss)

    def on_after_validation_epoch(self):
        log_dict = {
            'MSE': torch.tensor(self.val_losses).mean()
        }
        return log_dict


def compute_and_get_log_dict(metric, suffix=''):
    result_dict = metric.compute()
    metric_name = metric.__class__.__name__

    log_dict = {}

    for k, v in result_dict.items():
        name = f'{metric_name}{suffix}/{k}'
        log_dict[name] = v
    return log_dict