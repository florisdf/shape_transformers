from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import point_cloud_utils as pcu
import torch
from torch.utils.data import Dataset


TEST_SUBJECTS = [99, 283, 143, 38, 241, 236, 276, 202, 98, 254, 204, 163, 267, 194, 20, 23, 209, 105, 186, 343, 341, 363, 350]
BAD_SCANS = {
    261: [19],
    88: [19],
    79: [16, 17, 18, 19, 20],
    100: [0],
    125: [1, 4, 5],
    106: [20],
    362: [20],
    363: [1],
    345: [12],
    360: [6, 14],
    85: [2],
    292: [9],
    298: [23, 24, 25, 26]
}
NEUTRALS = {17: 10, 18: 10, 19: 10, 20: 10, 22: 9, 23: 12, 24: 10, 25: 10, 26: 10, 27: 10, 28: 10, 29: 10, 31: 10, 32: 10, 33: 10, 34: 10, 35: 10, 36: 10, 37: 9, 38: 10, 39: 10, 40: 1, 41: 1, 42: 1, 43: 1, 44: 1, 45: 1, 46: 1, 48: 1, 49: 1, 50: 1, 51: 1, 52: 1, 53: 1, 54: 1, 55: 1, 56: 1, 57: 1, 58: 1, 59: 1, 60: 1, 61: 1, 62: 1, 63: 2, 64: 24, 65: 1, 67: 1, 68: 1, 69: 1, 70: 1, 71: 1, 72: 1, 73: 1, 74: 2, 75: 1, 76: 1, 77: 1, 78: 1, 79: 1, 80: 1, 81: 23, 82: 1, 83: 1, 84: 0, 85: 1, 86: 1, 87: 1, 88: 1, 89: 1, 90: 1, 91: 1, 92: 1, 93: 0, 94: 1, 95: 1, 96: 1, 97: 1, 98: 2, 99: -1, 100: 1, 102: 1, 103: 1, 104: 1, 105: -1, 106: 1, 108: 1, 109: 1, 110: 1, 111: 1, 112: 1, 113: 1, 114: 1, 115: 1, 116: 2, 117: 1, 118: 1, 120: 1, 121: 1, 122: 1, 123: 1, 124: 1, 125: -1, 126: 1, 127: 1, 128: 1, 129: 1, 130: 1, 131: 1, 132: 1, 133: 1, 134: 22, 135: 1, 136: 1, 137: 1, 138: 1, 140: 2, 141: 1, 142: 1, 143: 1, 144: 1, 145: 1, 146: 1, 147: 3, 148: 1, 149: 1, 150: 1, 151: 1, 162: 1, 163: 1, 164: 1, 165: 1, 167: 1, 168: 1, 174: 1, 179: 1, 180: 1, 181: 1, 182: 1, 183: 1, 184: 1, 185: 1, 186: 1, 187: 1, 188: 1, 189: 1, 190: 1, 191: 1, 193: 1, 194: 4, 195: 2, 196: 1, 198: 1, 199: 1, 200: 1, 201: 1, 202: 1, 204: 1, 206: 1, 207: 1, 209: 1, 210: 1, 211: 1, 212: 1, 213: 1, 214: 1, 215: 1, 216: 1, 217: 1, 218: 1, 220: 1, 221: 1, 223: 1, 224: 1, 226: 1, 227: 1, 228: 1, 229: 1, 231: 1, 232: 1, 233: 1, 234: 1, 235: 1, 236: -1, 237: 1, 238: 1, 239: 1, 240: 1, 241: 1, 242: 1, 243: 1, 244: 2, 245: 1, 246: 1, 247: 1, 248: 1, 249: 1, 250: 1, 251: 1, 252: 1, 254: 1, 255: 1, 256: 1, 257: 1, 258: 1, 259: 1, 260: 1, 261: 3, 262: 1, 263: 1, 264: 1, 265: 1, 267: 0, 268: 1, 269: 1, 270: 1, 271: 1, 272: 1, 274: 1, 275: 1, 276: 1, 277: 1, 278: 1, 279: 1, 280: 1, 281: 1, 282: 1, 283: 0, 284: 1, 285: 1, 286: 1, 287: 1, 289: 1, 290: 1, 291: 1, 292: 1, 293: 1, 294: 1, 295: 1, 297: 1, 298: 1, 334: 12, 335: 8, 336: 10, 337: 10, 338: 10, 339: 10, 340: 10, 341: 1, 342: 10, 343: 10, 344: 10, 345: 13, 346: 10, 347: 11, 348: 11, 349: 10, 350: 10, 351: 10, 352: 10, 353: 10, 354: 10, 355: 1, 356: 10, 357: 10, 358: 10, 359: 10, 360: 11, 361: 10, 362: 10, 363: 10, 364: 11, 365: 13}
NEUTRALS_CLOSED = {17: 0, 18: 0, 19: 0, 20: 0, 22: 0, 23: -1, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 31: 0, 32: 0, 33: 0, 34: 0, 35: 0, 36: 0, 37: 0, 38: 0, 39: 0, 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 0, 46: 0, 48: 0, 49: 0, 50: 0, 51: 0, 52: 0, 53: 0, 54: 0, 55: 0, 56: 0, 57: 0, 58: 0, 59: 0, 60: 0, 61: 0, 62: 0, 63: 0, 64: 0, 65: 0, 67: 0, 68: 0, 69: 0, 70: 0, 71: 0, 72: 0, 73: 0, 74: 0, 75: 0, 76: 0, 77: 0, 78: 0, 79: 0, 80: 0, 81: 0, 82: 0, 83: 0, 84: 3, 85: 0, 86: 0, 87: 0, 88: 0, 89: 0, 90: 0, 91: 0, 92: 0, 93: 8, 94: 0, 95: 0, 96: 0, 97: 0, 98: 1, 99: 0, 100: -1, 102: 0, 103: 0, 104: 0, 105: -1, 106: 0, 108: 0, 109: 0, 110: 0, 111: 0, 112: 0, 113: 0, 114: 0, 115: 0, 116: 0, 117: 0, 118: 0, 120: 0, 121: 0, 122: 0, 123: 0, 124: 0, 125: 0, 126: 0, 127: 0, 128: 0, 129: 0, 130: 0, 131: 0, 132: 0, 133: 0, 134: 21, 135: 0, 136: 0, 137: 0, 138: 0, 140: 1, 141: 0, 142: 0, 143: 0, 144: 0, 145: 0, 146: 0, 147: 2, 148: 0, 149: 0, 150: 0, 151: 0, 162: 0, 163: 0, 164: 0, 165: 0, 167: 0, 168: 0, 174: 0, 179: 0, 180: 0, 181: 0, 182: 0, 183: 24, 184: 0, 185: 0, 186: 0, 187: 0, 188: 0, 189: 0, 190: 0, 191: 0, 193: 0, 194: 23, 195: 1, 196: 0, 198: 0, 199: 0, 200: 0, 201: 0, 202: 0, 204: 0, 206: 0, 207: 0, 209: 0, 210: 0, 211: 0, 212: 0, 213: 0, 214: 0, 215: 0, 216: 0, 217: 0, 218: 0, 220: 0, 221: 0, 223: 0, 224: 0, 226: 0, 227: 0, 228: 0, 229: 0, 231: 0, 232: 0, 233: 0, 234: 0, 235: 0, 236: 0, 237: 0, 238: 0, 239: 0, 240: 0, 241: 0, 242: 0, 243: 0, 244: 1, 245: 0, 246: 0, 247: 0, 248: 0, 249: 0, 250: 0, 251: 0, 252: 0, 254: 0, 255: 0, 256: 0, 257: 0, 258: 0, 259: 0, 260: 0, 261: 2, 262: 0, 263: 0, 264: 0, 265: 0, 267: -1, 268: 0, 269: 0, 270: 0, 271: 0, 272: 0, 274: 0, 275: 0, 276: 0, 277: 0, 278: 0, 279: 0, 280: 0, 281: 0, 282: 0, 283: 1, 284: 0, 285: 0, 286: 0, 287: 0, 289: 0, 290: 0, 291: 0, 292: 0, 293: 0, 294: 0, 295: 0, 297: 0, 298: 0, 334: 0, 335: -1, 336: 0, 337: 0, 338: 0, 339: 0, 340: 0, 341: 0, 342: 0, 343: 0, 344: 0, 345: 0, 346: 0, 347: 0, 348: 0, 349: 0, 350: 0, 351: 0, 352: 0, 353: 0, 354: 0, 355: 0, 356: 0, 357: 0, 358: 0, 359: 0, 360: 0, 361: 0, 362: 0, 363: 0, 364: 1, 365: 16}


class NPHMDataset(Dataset):
    def __init__(
        self,
        data_path: Path,
        subset: Literal['train', 'test'],
        scan_type: Literal['registration', 'scan', 'flame'] = 'registration',
        drop_bad: bool = True,
        n_subsample: int = None,
        subsample_seed: int = None,
    ):
        df = get_nphm_df(data_path / subset)
        df = df[df['subset'] == subset].reset_index(drop=True)

        if drop_bad:
            df = df[~df['is_bad']].reset_index(drop=True)

        subject_to_label = {
            subj: i
            for i, subj in enumerate(df['subject'].unique())
        }
        df['label'] = df['subject'].apply(lambda s: subject_to_label[s])

        self.mean_verts = torch.tensor(np.load('nphm_mean_vertices.npy'))
        self.scan_type = scan_type
        self.data_path = data_path
        self.df = df
        self.subject_to_label = subject_to_label

        self.refresh_vert_idxs(n_subsample, subsample_seed)

    def refresh_vert_idxs(self, n_subsample=None, seed=None):
        idxs = np.arange(self.mean_verts.shape[0])

        if n_subsample is not None:
            np.random.seed(seed)
            np.random.shuffle(idxs)
            idxs = idxs[:n_subsample]

        self.vert_idxs = torch.tensor(idxs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        scan_path = row[self.scan_type]
        label = row['label']

        v, f = pcu.load_mesh_vf(scan_path)
        v = torch.tensor(v)[self.vert_idxs]
        mean_verts = self.mean_verts[self.vert_idxs]

        v_off = v - mean_verts
        positions = torch.clone(mean_verts)

        return v_off, positions, label


def get_nphm_df(data_path):
    rows = []

    for subj_path in data_path.glob('[0-9][0-9][0-9]'):
        subject = int(subj_path.name)

        subset = (
            'test' if subject in TEST_SUBJECTS
            else 'train'
        )

        for f in subj_path.glob("*"):
            expression = int(f.name)

            try:
                is_bad = expression in BAD_SCANS[subject]
            except KeyError:
                is_bad = False

            try:
                is_neutral_open = NEUTRALS[subject] == expression
            except KeyError:
                is_neutral_open = False

            try:
                is_neutral_closed = NEUTRALS_CLOSED[subject] == expression
            except KeyError:
                is_neutral_closed = False

            subj_expr_path = subj_path / f"{expression:03d}"

            rows.append({
                'subject': subject,
                'expression': expression,
                'is_bad': is_bad,
                'subset': subset,
                'scan': subj_expr_path / 'scan.ply',
                'flame': subj_expr_path / 'flame.ply',
                'registration': subj_expr_path / 'registration.ply',
                'is_neutral_open': is_neutral_open,
                'is_neutral_closed': is_neutral_closed,
            })

    return pd.DataFrame(rows)
