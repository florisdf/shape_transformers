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
        loss_dict = {}
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = self.model(positions, verts)
        else:
            pred_verts = self.model(positions, verts)

        loss = torch.sqrt(F.mse_loss(pred_verts, verts))
        loss_dict['L2'] = loss

        if self.model.disentangle_style:
            assert self.model.id_classifier is not None
            pred_logits = self.model.id_classifier(pred_id_embs)

            loss_cel = F.cross_entropy_loss(pred_logits, labels)
            loss_dict['CEL'] = loss_cel
            loss += loss_cel

        loss_dict['TotalLoss'] = loss

        return loss, loss_dict

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        pass

    def on_validation_step(self, batch):
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = self.model(positions, verts)
        else:
            pred_verts = self.model(positions, verts)

        loss = torch.sqrt(F.mse_loss(pred_verts, verts))
        self.val_losses.append(loss)

    def on_after_validation_epoch(self):
        log_dict = {
            'ValLoss/L2': torch.tensor(self.val_losses).mean()
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
