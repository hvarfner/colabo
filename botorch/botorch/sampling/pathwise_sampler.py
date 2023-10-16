from __future__ import annotations

import torch
from botorch.posteriors import Posterior
from botorch.posteriors.deterministic import DeterministicPosterior
from botorch.sampling.base import MCSampler
from torch import Tensor


class PathwiseSampler(MCSampler):

    def __init__(
        self,
        sample_shape: torch.Size,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            sample_shape=sample_shape,
            seed=seed,
            **kwargs,
        )

    def forward(self, posterior: PathPosterior):
        # The matheron samples are not quite giving the desired shape just yet
        return posterior.rsample(sample_shape=self.sample_shape).unsqueeze(-1)
