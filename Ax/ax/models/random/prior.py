from typing import Optional

import numpy as np
from ax.models.random.base import RandomModel
from scipy.stats import uniform
from botorch.utils.prior import UserPrior
from torch import Tensor


class PriorGenerator(RandomModel):
    """This class specifies a uniform random generation algorithm.
    As a uniform generator does not make use of a model, it does not implement
    the fit or predict methods.
    Attributes:
        seed: An optional seed value for the underlying PRNG.
    """

    def __init__(self, prior: UserPrior, deduplicate: bool = False, seed: Optional[int] = None, generate_default: bool = True) -> None:
        super().__init__(deduplicate=deduplicate, seed=seed)
        self.prior = prior
        self.generate_default = generate_default
    
    def _gen_samples(self, n: int, tunable_d: int) -> np.ndarray:
        """Generate samples from the scipy uniform distribution.
        Args:
            n: Number of samples to generate.
            tunable_d: Dimension of samples to generate.
        Returns:
            samples: An (n x d) array of random points.
        """
        if  self.generate_default:
            return self.prior._default
            
        return self.prior._sample(num_samples=n)  # pyre-ignore
