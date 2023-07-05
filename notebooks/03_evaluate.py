#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import sys; sys.path.append('..')
import os

import argparse
import cv2
import numpy as np
from PIL import Image
import pyrender
from scipy.spatial.transform import Rotation
import torch
from trimesh import Trimesh
from tqdm import tqdm

from shape_transformers.model.shape_transformer import ShapeTransformer
from train import get_data_loaders
from utils.wandb_loader import get_run_results


os.environ['PYOPENGL_PLATFORM'] = 'egl'


def get_pred_shape(model, true_rel_verts, positions):
    with torch.no_grad():
        pred_rel_verts = model(positions[None, ...],
                               true_rel_verts[None, ...]).squeeze()
    return pred_rel_verts


def get_interpolated_shape(model, true_rel_verts_1, true_rel_verts_2, positions, alpha):
    with torch.no_grad():
        shape_code_1 = model.encoder(
            positions[None, ...],
            true_rel_verts_1[None, ...]
        )
        shape_code_2 = model.encoder(
            positions[None, ...],
            true_rel_verts_2[None, ...]
        )
        shape_code = (
            (1 - alpha) * shape_code_1
            + alpha * shape_code_2
        )
        pred_rel_verts = model.decoder(
            positions[None, ...],
            shape_code
        ).squeeze()
    return pred_rel_verts


def rel_to_abs_verts(rel_verts, ds):
    shape_norm = ds.transform
    return shape_norm.inverse(rel_verts)


def get_camera_pose(x, y, z, pitch, yaw, roll):
    euler_angles = [pitch, yaw, roll]
    rotation = Rotation.from_euler('xyz', np.radians(euler_angles)).as_matrix()
    rotation
    translation = np.array([x, y, z])
    pose_world = np.vstack([
        np.hstack([rotation, translation[:, None]]),
        np.array([0, 0, 0, 1])
    ])
    pose_cam = np.linalg.inv(pose_world)
    return pose_cam


def render_mesh(vertices, triangles):
    tm = Trimesh(vertices, triangles)
    mesh = pyrender.Mesh.from_trimesh(tm)

    # compose scene
    scene = pyrender.Scene(ambient_light=[.1, .1, .3], bg_color=[255, 255, 255])
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 6.0)
    light = pyrender.DirectionalLight(color=[1,1,1], intensity=2e3)

    scene.add(mesh, pose=np.eye(4))
    scene.add(light, pose=np.eye(4))

    cam_pose = get_camera_pose(
        x=0, y=.1, z=-2.5,
        pitch=0, yaw=0, roll=0
    )
    scene.add(camera, pose=cam_pose)

    # render scene
    r = pyrender.OffscreenRenderer(512, 512)

    try:
        color, depth = r.render(scene)
        return color
    except:
        print('Render failed')


def get_model_and_data(
    run_id,
):
    df = get_run_results(run_id=run_id, lazy=False)
    row = df.iloc[0]
    token_size = row['token_size']
    disentangle_style = row['disentangle_style']

    model = ShapeTransformer(token_size, disentangle_style)
    ckpt = torch.load(f'../ckpts/{run_id}_best.pth')
    model.load_state_dict(ckpt)
    model.eval()

    dl_train, dl_val, dl_test = get_data_loaders(
        row.data_path, row.scan_type, not row.keep_bad_scans, row.n_verts_subsample,
        row.subsample_seed, row.k_fold_num_folds, row.k_fold_val_fold, row.k_fold_seed,
        batch_size=1, val_batch_size=1, num_workers=8
    )
    ds = dl_val.dataset
    ds.transform = ds.transform.transforms[0]

    return model, ds


def select_sample(ds, idx, device):
    true_rel_verts, positions, triangles, label = ds[idx]
    true_rel_verts = true_rel_verts.to(device)
    positions = positions.to(device)
    return true_rel_verts, positions, triangles, label


def check_label(ds, label):
    if label not in ds.df.label.values:
        raise ValueError(f'Subject "{label}" is not present in the validation dataset. Please choose from {", ".join([str(l) for l in ds.df.label.unique()])}.')


def get_expression_label_idx(ds, label, expression):
    check_label(ds, label)
    df_label = ds.df[ds.df['label'] == label]
    if expression not in df_label.expression.values:
        raise ValueError(f'Subject "{label}" has no expression "{expression}" in the validation dataset. Please choose from {", ".join([str(l) for l in sorted(df_label.expression.unique())])}.')
    return df_label[df_label['expression'] == expression].index[0]


def render_true_pred(
    model, ds, label, expression, device,
    render_path_true, render_path_pred
):
    idx = get_expression_label_idx(ds, label, expression)
    true_rel_verts, positions, triangles, _ = select_sample(ds, idx, device)
    pred_rel_verts = get_pred_shape(model, true_rel_verts, positions)

    true_verts = rel_to_abs_verts(true_rel_verts.cpu(), ds)
    pred_verts = rel_to_abs_verts(pred_rel_verts.cpu(), ds)

    true_im_arr = render_mesh(true_verts, triangles)
    pred_im_arr = render_mesh(pred_verts, triangles)

    Image.fromarray(true_im_arr).save(render_path_true)
    Image.fromarray(pred_im_arr).save(render_path_pred)


def render_expression_video(
    model, ds, label, device,
    frames_per_second, seconds_per_expression,
    render_path
):
    check_label(ds, label)
    df_label = ds.df[ds.df.label == label]
    idxs = [idx for idx, _ in df_label.iterrows()]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = None

    frames_per_expression = int(frames_per_second * seconds_per_expression)
    for idx1, idx2 in tqdm(list(zip(idxs, idxs[1:])), leave=False):
        true_rel_verts_1, positions, triangles, _ = select_sample(ds, idx1, device)
        true_rel_verts_2, _, _, _ = select_sample(ds, idx2, device)
        d_alpha = 1 / frames_per_expression

        for i in range(frames_per_expression):
            alpha = i * d_alpha
            pred_rel_verts = get_interpolated_shape(
                model, true_rel_verts_1, true_rel_verts_2,
                positions, alpha
            )
            pred_verts = rel_to_abs_verts(pred_rel_verts.cpu(), ds)
            pred_im_arr = render_mesh(pred_verts, triangles)
            if writer is None:
                w, h, c = pred_im_arr.shape
                writer = cv2.VideoWriter(str(render_path), fourcc, frames_per_second, (w, h))
            writer.write(pred_im_arr[..., ::-1])

    writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--device',
        help='Torch device',
        default='cuda'
    )
    parser.add_argument(
        '--run',
        help='Wandb run ID',
        default='tsd0f0nv',
    )
    parser.add_argument(
        '--subject',
        help='The subject (person) to render.',
        type=int
    )
    parser.add_argument(
        '--expression',
        help='The expression to select for the chosen subject.',
        type=int
    )
    parser.add_argument(
        '--render_expression_video',
        help='If set, render a video interpolating between all expressions of the chosen subject.',
        action='store_true'
    )
    parser.add_argument(
        '--render_all_expression_videos',
        help='If set, render all videos interpolating between all expressions of each subject.',
        action='store_true'
    )
    parser.add_argument(
        '--render_true_pred',
        help='If set, render the true and predicted shapes of the selected subject and expression.',
        action='store_true'
    )
    parser.add_argument(
        '--render_all_true_preds',
        help='If set, render all true and predicted shapes of each subject and expression.',
        action='store_true'
    )
    parser.add_argument(
        '--frames_per_second',
        help='FPS of the video to render.',
        type=float,
        default=30
    )
    parser.add_argument(
        '--seconds_per_expression',
        help='The number of seconds per expression.',
        type=float,
        default=0.5,
    )
    parser.add_argument(
        '--out_dir',
        help='The directory to save the created renders.',
        default='renders'
    )

    args = parser.parse_args()

    model, ds = get_model_and_data(args.run)
    model = model.to(args.device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subj, expr = args.subject, args.expression

    if args.render_true_pred:
        render_true_pred(
            model, ds, subj, expr, args.device,
            out_dir / f'{subj}_{expr}_true.jpg',
            out_dir / f'{subj}_{expr}_pred.jpg'
        )
    if args.render_expression_video:
        render_expression_video(
            model, ds, subj, args.device,
            args.frames_per_second,
            args.seconds_per_expression,
            out_dir / f'{subj}_expressions.mp4'
        )
    if args.render_all_expression_videos:
        for subj in tqdm(ds.df.label.unique()):
            render_expression_video(
                model, ds, subj, args.device,
                args.frames_per_second,
                args.seconds_per_expression,
                out_dir / f'{subj}_expressions.mp4'
            )
    if args.render_all_true_preds:
        for (subj, expr), _ in tqdm(list(ds.df.groupby(['label', 'expression']))):
            render_true_pred(
                model, ds, subj, expr, args.device,
                out_dir / f'{subj}_{expr}_true.jpg',
                out_dir / f'{subj}_{expr}_pred.jpg'
            )