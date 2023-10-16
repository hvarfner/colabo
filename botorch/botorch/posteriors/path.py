from __future__ import annotations

from typing import Optional
from warnings import warn

import torch
from botorch.posteriors.posterior import Posterior
from torch import Tensor


class PathPosterior(Posterior):

    def __init__(
        self, 
        paths: ModuleList, 
        X: Tensor, 
        posterior_transform: Optional[PosteriorTransform] = None, 
        subset_indices: Optional[Tensor] = None
    ):

        self.out_dtype = X.dtype
        X = X.to(paths.paths.prior_paths.weight.dtype)
        self.X = X
        self.paths = paths
        if X.ndim == 2:
            X_query = X.unsqueeze(-2)
        else:
            X_query = X.clone()
        self.output = self.paths(X_query.unsqueeze(-3), subset=subset_indices)

        if X.ndim == 2:
            self.output = self.output

    @property
    def device(self) -> torch.device:
        r"""The torch device of the distribution."""
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        r"""The torch dtype of the distribution."""
        return self._dtype

    def rsample(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        #assert sample_shape == self.output.shape[1:2], f'sample_shape is {sample_shape}, should be {self.output.shape[1]}'
        return self.output.transpose(0, 1).to(self.out_dtype)

    def rsample_from_base_samples(self, sample_shape: Optional[torch.Size] = None) -> Tensor:
        #assert sample_shape == self.output.shape[1:2], f'sample_shape is {sample_shape}, should be {self.output.shape[1]}'
        return self.output.transpose(0, 1).to(self.out_dtype)

    @property
    def mean(self):
        # TODO check whether this is the right dimension to evarage over
        return self.output.mean(dim=-2).to(self.out_dtype)

    @property
    def variance(self):
        return self.output.var(dim=-2).to(self.out_dtype)
