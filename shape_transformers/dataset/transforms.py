import numpy as np
import torch


class ShapePositionNormalize:
    def __init__(
        self,
        vertex_mean,
        vertex_std,
        eps=1e-6
    ):
        self.mean_shape = torch.tensor(vertex_mean)
        self.std_shape = torch.tensor(vertex_std)
        self.eps = eps

        self.mean_shape_mean = self.mean_shape.mean()
        self.mean_shape_std = self.mean_shape.std()

        self.normed_positions = (
            (self.mean_shape - self.mean_shape_mean)
            / (self.mean_shape_std + self.eps)
        )

    def inverse(self, normalized_vertices):
        return (
            normalized_vertices * (self.std_shape + self.eps)
            + self.mean_shape
        )

    def __call__(self, vertices, *other):
        v_norm = (
            (vertices - self.mean_shape)
            / (self.std_shape + self.eps)
        )
        return v_norm, torch.clone(self.normed_positions), *other


class SubsampleShape:
    def __init__(
        self,
        n_subsample: int = None,
        seed: int = None
    ):
        self.n_subsample = n_subsample
        self.seed = seed
        self.tot_samples = None
        self.vert_idxs = None

    def refresh_vert_idxs(self):
        assert self.tot_samples is not None
        idxs = np.arange(self.tot_samples)

        if self.n_subsample is not None:
            np.random.seed(self.seed)
            np.random.shuffle(idxs)
            idxs = idxs[:self.n_subsample]

        self.vert_idxs = torch.tensor(idxs)

    def __call__(self, vertices, positions, *other):
        if self.tot_samples is None:
            assert len(vertices) == len(positions)
            self.tot_samples = len(vertices)
        if self.vert_idxs is None:
            self.refresh_vert_idxs()

        return (
            vertices[self.vert_idxs],
            positions[self.vert_idxs],
            *other
        )


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, *x):
        for tfm in self.transforms:
            x = tfm(*x)
        return x
