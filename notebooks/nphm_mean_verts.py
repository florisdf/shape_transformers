#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sys; sys.path.append('..')


from nphm_dataset import NPHMDataset
from pathlib import Path

data_path = Path('/apollo/datasets/NPHM/train/')
ds_train = NPHMDataset(data_path, subset='train')


# In[ ]:


from torch.utils.data import DataLoader


# In[ ]:


batch_size = 128

dl_train = DataLoader(
    ds_train,
    batch_size=batch_size,
    num_workers=8
)


# In[ ]:


from tqdm.notebook import tqdm
import torch

pos_means = []
v_off_means = []
v_off_vars = []

for v_off, pos, ids in tqdm(dl_train):
    pos = pos.to('cuda')
    v_off = v_off.to('cuda')
    pos_means.append(
        pos.mean(dim=0)
    )
    v_off_means.append(
        v_off.mean(dim=0)
    )
    v_off_vars.append(
        v_off.std(dim=0).pow(2)
    )
    last_batch_size = len(v_off)

pos_means = torch.stack(pos_means)
v_off_means = torch.stack(v_off_means)
v_off_vars = torch.stack(v_off_vars)

def agg_means(means):
    return (torch.sum(means[:-1, ...], dim=0) + (last_batch_size / batch_size) * means[-1, ...])/len(means)


pos_mean = agg_means(pos_means)
v_off_mean = agg_means(v_off_means)
v_off_std = torch.sqrt(agg_means(v_off_vars))


# In[ ]:


import numpy as np

np.save('nphm_mean_vertices.npy', pos_mean.cpu().numpy())
np.save('nphm_mean_offsets.npy', v_off_mean.cpu().numpy())
np.save('nphm_std_offsets.npy', v_off_std.cpu().numpy())

