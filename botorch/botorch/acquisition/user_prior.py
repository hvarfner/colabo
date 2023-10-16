from abc import ABC
import math
from typing import Any, Callable, Optional


import numpy as np
import torch
from torch import Tensor
import torch.distributions as dist
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.model import Model
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.models.utils import check_no_nans
from botorch.utils import t_batch_mode_transform
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement, qProbabilityOfImprovement
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform
)
from botorch.utils.prior import UserPrior

r"""
(User-specified) Prior-weighted Acquisition functions - biasing the
acquisition function with a user-specified a prior belief over the 
location of the optimum.

References:

.. [Hvarfner2022pibo]
    C. Hvarfner, D. Stoll, A. Souza, M. Lindauer, F. Hutter, L. Nardi. 
    $\pi$BO: Augmenting Acquisition Functions with User Beliefs for 
    Bayesian Optimization. International Conference on Learning 
    Representations, 2022.
"""

class PriorAcquisitionFunction(AcquisitionFunction):

    def __init__(
        self,
        user_prior: UserPrior,
        raw_acqf: AcquisitionFunction,
        raw_acqf_kwargs: Any = {},
        decay_beta: float = 10.0,
        prior_floor: float = 1e-12,
        log_acq_floor: float = 1e-30,
        nonneg_acq: bool = True,
        **kwargs
    ):

        super().__init__(model=raw_acqf_kwargs['model'], **kwargs)
        # TODO

        self.user_prior = user_prior
        self.raw_acqf = raw_acqf(**raw_acqf_kwargs)
        self.decay_factor = decay_beta / \
                (len(self.model.train_targets))
        self.nonneg_acq = nonneg_acq
        self.prior_floor = torch.Tensor([prior_floor])
        self.acq_floor = torch.Tensor([log_acq_floor])
        
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        # If the acq is nonnegative, we gain numerical stability
        # by evaluating in logspace (since the  or may give us)
        # unstable, small values
        raw_value = self.raw_acqf(X=X)
        log_prior_value = torch.clamp_min(
        self.user_prior.forward(X), torch.log(self.prior_floor))

        if self.nonneg_acq:
            return log_prior_value * self.decay_factor + torch.log(torch.clamp_min(raw_value, self.acq_floor))

        return torch.pow(torch.exp(log_prior_value), self.decay_factor) * raw_value


class PriorqExpectedImprovement(PriorAcquisitionFunction):

    def __init__(
        self,
        user_prior: UserPrior,
        raw_acqf_kwargs: Any = {},
        decay_beta: float = 10.0,
        custom_decay_factor: float = 0.0,
        prior_floor: float = 1e-12,
        log_acq_floor: float = 1e-30,
        nonneg_acq: bool = True,
        **kwargs
    ):
        super().__init__(
            user_prior=user_prior,
            raw_acqf=qExpectedImprovement,
            raw_acqf_kwargs=raw_acqf_kwargs,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            log_acq_floor=log_acq_floor,
            nonneg_acq=nonneg_acq,
            **kwargs
        )


class PiBO(PriorAcquisitionFunction):

    def __init__(
        self,
        user_prior: UserPrior,
        raw_acqf_kwargs: Any = {},
        decay_beta: float = 10.0,
        custom_decay_factor: float = 0.0,
        prior_floor: float = 1e-12,
        log_acq_floor: float = 1e-30,
        nonneg_acq: bool = True,
        **kwargs
    ):
        super().__init__(
            user_prior=user_prior,
            raw_acqf=qNoisyExpectedImprovement,
            raw_acqf_kwargs=raw_acqf_kwargs,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            log_acq_floor=log_acq_floor,
            nonneg_acq=nonneg_acq,
            **kwargs
        )


class PriorqProbabilityOfImprovement(PriorAcquisitionFunction):

    def __init__(
        self,
        user_prior: UserPrior,
        raw_acqf_kwargs: Any = {},
        decay_beta: float = 10.0,
        custom_decay_factor: float = 0.0,
        prior_floor: float = 1e-12,
        log_acq_floor: float = 1e-30,
        nonneg_acq: bool = True,
        **kwargs
    ):
        super().__init__(
            user_prior=user_prior,
            raw_acqf=qProbabilityOfImprovement,
            raw_acqf_kwargs=raw_acqf_kwargs,
            decay_beta=decay_beta,
            prior_floor=prior_floor,
            log_acq_floor=log_acq_floor,
            nonneg_acq=nonneg_acq,
            **kwargs
        )
