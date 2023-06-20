import torch
import torch.nn.functional as F
import wandb


class TrainingSteps:
    def __init__(
        self,
        model,
        max_num_3d_logs=5
    ):
        self.model = model

        self.val_losses = []
        self.test_losses = []
        self.val_point_clouds = []
        self.test_point_clouds = []
        self.max_num_3d_logs = max_num_3d_logs

    def on_before_training_epoch(self):
        pass

    def on_training_step(self, batch):
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = self.model(positions, verts)
        else:
            pred_verts = self.model(positions, verts)

        loss = torch.sqrt(F.mse_loss(pred_verts, verts))
        log_dict = {
            'L2': loss,
        }

        return loss, log_dict

    def on_after_training_epoch(self):
        return {}

    def on_before_validation_epoch(self):
        pass

    def on_before_test_epoch(self):
        pass

    def on_evaluation_step(self, batch, batch_idx, point_clouds, losses):
        verts, positions, labels = batch
        if self.model.disentangle_style:
            pred_verts, pred_id_embs = self.model(positions, verts)
        else:
            pred_verts = self.model(positions, verts)

        if batch_idx == 0:
            for v, p, l, pred, _ in zip(
                verts, positions, labels, pred_verts,
                range(self.max_num_3d_logs)
            ):
                v = wandb.Object3D((v + p).cpu().numpy())
                pred = wandb.Object3D((pred + p).cpu().numpy())
                point_clouds.extend([
                    {f"{int(l.cpu())}_true_points": v},
                    {f"{int(l.cpu())}_pred_points": pred},
                ])

        loss = torch.sqrt(F.mse_loss(pred_verts, verts))
        losses.append(loss)

    def on_validation_step(self, batch, batch_idx):
        return self.on_evaluation_step(batch, batch_idx, self.val_point_clouds, self.val_losses)

    def on_test_step(self, batch, batch_idx):
        return self.on_evaluation_step(batch, batch_idx, self.test_point_clouds, self.test_losses)

    def on_after_evaluation_epoch(self, point_clouds, losses):
        log_dict = {
            'L2': torch.tensor(losses).mean()
        }
        for d in point_clouds:
            log_dict.update(d)

        point_clouds.clear()
        losses.clear()

        return log_dict

    def on_after_validation_epoch(self):
        return self.on_after_evaluation_epoch(self.val_point_clouds, self.val_losses)

    def on_after_test_epoch(self):
        return self.on_after_evaluation_epoch(self.test_point_clouds, self.test_losses)


def compute_and_get_log_dict(metric, suffix=''):
    result_dict = metric.compute()
    metric_name = metric.__class__.__name__

    log_dict = {}

    for k, v in result_dict.items():
        name = f'{metric_name}{suffix}/{k}'
        log_dict[name] = v
    return log_dict
