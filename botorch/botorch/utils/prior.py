from __future__ import annotations
from abc import ABC, abstractmethod

import math
from typing import Any, Callable, Optional


import numpy as np
import torch
from torch import Tensor
import torch.distributions as dist
from botorch.utils.transforms import normalize, standardize, unnormalize
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform
)
from botorch.test_functions import *
from botorch.sampling.pathwise import MatheronPath
from botorch.utils.sampling import optimize_posterior_samples


class UserPrior:
    pass


class UserPriorLocation:

    def __init__(self, bounds, prior_floor: float = 1e-12, dtype: torch.dtype = torch.float, seed: int = 42):
        self.bounds = bounds
        self.norm_bounds = Tensor([(0.0, 1.0) for dim in range(bounds.shape[1])]).T
        self.dim = bounds.shape[1]
        self.prior_floor = prior_floor
        self.dtype = dtype
        self.seed = seed

    def register_maxval(self, optval_prior: UserPriorValue):
        self.optval_prior = optval_prior

    def compute_logprobs(self, matheron_paths: MatheronPath, raw_samples: int = 2 ** 11, **kwargs):
        """Given function samples, the (unnormalized) probability of 
        the sample's occurance is computed.

        Args:
            matheron_paths (MatheronPath): paths to be queried for 
            their unnormalized probability.
        """
        import time
        #print('Starting opt')
        start = time.time()
        suggestions = self._sample(num_samples=raw_samples)
        if hasattr(self, 'default'):
            more_suggestions = self._default + \
                torch.rand(size=(raw_samples, self.default.shape[0])) * 0.01
            suggestions = torch.cat((suggestions, more_suggestions))
            more_suggestions = self._default + \
                torch.rand(size=(raw_samples, self.default.shape[0])) * 0.025
            suggestions = torch.cat((suggestions, more_suggestions))
        self.optimal_inputs, self.optimal_outputs = optimize_posterior_samples(
            matheron_paths, bounds=self.norm_bounds, raw_samples=raw_samples, candidates=suggestions, num_restarts=0)

        #print('Finihsed', time.time() - start)
        # breakpoint()
        logprobs = self.forward(self.optimal_inputs)
        if self.optval_prior is not None:
            logprobs_output = self.optval_prior.evaluate(self.optimal_outputs)
            logprobs = logprobs + logprobs_output.T
        return logprobs

    def get_optima(self):
        return self.optimal_inputs, self.optimal_outputs

    def compute_norm_probs(
            self,
            matheron_paths: MatheronPath,
            decay_factor: Optional[Union[Tensor, int, float]] = 1.0,
            prior_floor: Optional[Union[Tensor, int, float]] = 0.0,
            **kwargs):
        """Computing the actual probability (although still unproper) from the logged
        ones, for numerical stability.

        Args:
            matheron_paths (MatheronPath): paths to be queried for 
            their unnormalizedprobability.

        Returns:
            Tensor: (N,) shaped tensor of probabilities. 
        """
        logprobs = self.compute_logprobs(matheron_paths, **kwargs)
        logprobs_norm = logprobs - logprobs.max()
        probs = torch.exp(logprobs_norm)
        # breakpoint()
        # These are split relatively - how can we make that smarter?
        decay_probs = torch.pow(probs, decay_factor).clamp_min(prior_floor)

        norm_probs = len(probs) * decay_probs / decay_probs.sum()
        return norm_probs

    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """Acts as the joint pdf function in the unnormalized space. As such, the input 
        enters in the original space defined by self.bounds.

        Args: 
            X (torch.Tensor): The unnormalized input tensor

        Returns:
            torch.Tensor: The log probability density of each sample in the original space.
        """
        pass

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Acts as the joint pdf function in the normalized space.

        Args:
            X (torch.Tensor): The normalized input tensor in range [0, 1]

        Returns:
            torch.Tensor: The probability density of each sample.
        """
        # TODO Loop through a list of priors?

        # TODO check whether squeezing is actually the right thing to do
        return self.evaluate(X).squeeze(1)

    @abstractmethod
    def sample(self, num_samples: int = 1):
        """Retrieves samples in the actual, true search space

        Args:
            num_samples ([type]): Number of samples
        """
        pass

    def _sample(self, num_samples: int = 1):
        """Retrieves samples in the normalized search space

        Args:
            num_samples ([type]): [description]
        """
        return normalize(self.sample(num_samples=num_samples), self.bounds)


class DefaultPrior(UserPriorLocation):
    # TODO accound for discrete / categorical
    """A prior where defaults (and possibly the standard deviation / confidence)
    is passed in and a multivariate gaussian is formed around these
    """

    def __init__(self, bounds, parameter_defaults: torch.Tensor, confidence: Optional[float, list, torch.Tenor] = 0.25, spread_dim: bool = True):

        super().__init__(bounds)
        self.priors_list = []
        self.parameter_defaults = normalize(parameter_defaults, bounds)

        self.spread_dim = spread_dim
        assert len(
            parameter_defaults.shape) == 1, f"Parameter defaults needs to be 2-dimensional, of shape {self.dim},"

        if isinstance(confidence, float):
            confidence = torch.ones(self.dim) * confidence
            unnorm_confidence = confidence
        elif isinstance(confidence, (list, torch.Tensor)):
            unnorm_confidence = (
                self.bounds[1, :] - self.bounds[0, :]) * torch.Tensor(confidence)
        else:
            raise TypeError(
                f"Confidence must be either float or a tensor of length {self.dim}.")
        # Not making it multivariate normal since it may contain many different parameter types
        self.norm_factors = []
        if self.spread_dim:
            spread_factor = math.sqrt(self.dim)
        else:
            spread_factor = 1

        for default, conf in zip(self.parameter_defaults, unnorm_confidence):
            unnorm_distr = dist.Normal(default, conf)
            self.norm_factors.append(
                (unnorm_distr.cdf(Tensor([1])) - unnorm_distr.cdf(Tensor([0]))))
            self.priors_list.append(unnorm_distr)

    @property
    def default(self):
        return unnormalize(self.parameter_defaults, self.bounds)

    @property
    def _default(self):
        return self.parameter_defaults

    def sample(self, num_samples):
        output = torch.empty(torch.Size([num_samples, self.dim]))
        for dim in range(self.dim):
            output[:, dim] = self.priors_list[dim].rsample(
                sample_shape=torch.Size([num_samples]))

        return unnormalize(output, self.bounds)

    def evaluate(self, X):
        log_prob = torch.zeros_like(X)
        for dim in range(self.dim):
            log_prob[:, :, dim] = self.priors_list[dim].log_prob(
                X[:, :, dim]) - math.log(self.norm_factors[dim])
        return torch.sum(log_prob, axis=-1)


class PreferencePrior(UserPriorLocation):
    # TODO accound for discrete / categorical
    """A prior where defaults (and possibly the standard deviation / confidence)
    is passed in and a multivariate gaussian is formed around these
    """

    def __init__(self, bounds, better_configs: torch.Tensor, worse_configs: torch.Tensor, confidence: Optional[float, list, torch.Tenor] = 0.8):

        super().__init__(bounds)
        self.priors_list = []
        self.better_configs = unnormalize(better_configs, self.bounds)
        self.worse_configs = unnormalize(worse_configs, self.bounds)

    @property
    def _default(self):
        return self.better_configs[0]

    def sample(self, num_samples):
        if len(self.parameter_defaults) < num_samples:
            raise ValueError(
                f'Requesting too many samples. PreferencePrior has {len(self.better_configs)} better configs, requested {num_samples}')
        sample_index = np.random.choice(
            len(self.better_configs), num_samples, replace=False)
        return self.better_configs[sample_index]

    def evaluate(self, X):
        log_prob = torch.zeros_like(X)
        for dim in range(self.dim):
            log_prob[:, :, dim] = self.priors_list[dim].log_prob(X[:, :, dim])
        return torch.sum(log_prob, axis=-1)


class UserPriorValue(UserPrior):

    def __init__(self, prior_floor: float = 1e-12, dtype: torch.dtype = torch.float, seed: int = 42):
        pass

    def setup(self, Y_normalized, mean, std):
        self.Y_unnormalized = Y_normalized * std + mean
        self.mean = mean
        self.std = std

    def _unnormalize(self, Y):
        return Y * self.std + self.mean

    def evaluate(self, Y_opt):
        pass

    @t_batch_mode_transform()
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Acts as the joint pdf function in the normalized space.

        Args:
            X (torch.Tensor): The normalized input tensor in range [0, 1]

        Returns:
            torch.Tensor: The probability density of each sample.
        """
        # TODO check shape
        return self.evaluate(Y)


class UserPriorHardMaxValue(UserPriorValue):

    def __init__(self, maxopt_value: float = None, minopt_value: float = None, prior_floor: float = 1e-12, dtype: torch.dtype = torch.float, seed: int = 42):
        super().__init__()
        self.minopt_value = minopt_value
        self.maxopt_value = maxopt_value
        self.prior_floor = prior_floor

    def evaluate(self, Y_opt):
        Y_unnormalized = self._unnormalize(Y_opt)
        y_probs = torch.ones_like(Y_unnormalized)

        if self.minopt_value is not None:
            y_probs = y_probs * Y_unnormalized > self.minopt_value
        if self.maxopt_value is not None:
            y_probs = y_probs * Y_unnormalized < self.maxopt_value

        return torch.log(y_probs + self.prior_floor)


class UserPriorMaxValue(UserPriorValue):

    def __init__(self, parameter_default: float, confidence: float, prior_floor: float = 1e-12, dtype: torch.dtype = torch.float, seed: int = 42):
        super().__init__()
        self.prior_dist = dist.Normal(torch.Tensor(
            [parameter_default]), torch.Tensor([confidence]))
        self.prior_floor = prior_floor

    def evaluate(self, Y_opt):
        Y_unnormalized = self._unnormalize(Y_opt)
        # If all are zero check
        y_probs = self.prior_dist.log_prob(Y_unnormalized)
        return y_probs + self.prior_floor
