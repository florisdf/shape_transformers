from copy import deepcopy

import torch
import torch.nn.functional as F
import wandb


class TrainingSteps:
    def __init__(
        self,
        model,
        cel_weight
    ):
        self.model = model
        self.cel_weight = cel_weight
        self.val_losses = []
        self.gallery_embs = []
        self.gallery_labels = []
        self.query_scores = []
        self.query_labels = []

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

            loss_cel = F.cross_entropy(pred_logits, labels)
            loss_dict['CEL'] = loss_cel
            loss = loss + self.cel_weight * loss_cel

        loss_dict['TotalLoss'] = loss

        return loss, loss_dict

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        pass

    def on_validation_gallery_step(self, batch):
        verts, positions, labels = batch
        pred_verts, pred_id_embs = self.model(positions, verts)
        self.gallery_embs.append(pred_id_embs)
        self.gallery_labels.append(labels)

    def on_validation_query_step(self, batch):
        gallery_embs = torch.cat(self.gallery_embs)
        gallery_labels = torch.cat(self.gallery_labels)

        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = self.model(positions, verts)
        else:
            pred_verts = self.model(positions, verts)

        self.query_scores.append(pred_id_embs @ gallery_embs.T)
        self.query_labels.append(labels)

        loss = torch.sqrt(F.mse_loss(pred_verts, verts))
        self.val_losses.append(loss)

    def on_after_validation_epoch(self):
        gallery_embs = torch.cat(self.gallery_embs)
        gallery_labels = torch.cat(self.gallery_labels)
        query_scores = torch.cat(self.query_scores)
        query_labels = torch.cat(self.query_labels)

        pred_idxs = query_scores.argmax(dim=1)
        pred_labels = gallery_labels[pred_idxs]

        log_dict = {
            'Val/L2': torch.tensor(self.val_losses).mean(),
            'Val/Acc': (pred_labels == query_labels).sum() / len(query_labels),
        }

        self.gallery_embs = []
        self.gallery_labels = []
        self.query_scores = []
        self.query_labels = []

        return log_dict


def compute_and_get_log_dict(metric, suffix=''):
    result_dict = metric.compute()
    metric_name = metric.__class__.__name__

    log_dict = {}

    for k, v in result_dict.items():
        name = f'{metric_name}{suffix}/{k}'
        log_dict[name] = v
    return log_dict
