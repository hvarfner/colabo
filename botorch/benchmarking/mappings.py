from botorch.test_functions import *

from ax.modelbridge.registry import Models

from benchmarking.pd1_task import PD1Function
from botorch.acquisition import (
    qMaxValueEntropy,
    qKnowledgeGradient,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.analytic import (
    AnalyticThompsonSampling,
    DiscreteThompsonSampling
)

from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.joint_entropy_search import (
    qJointEntropySearch,
    qExploitJointEntropySearch,
)
from botorch.utils.prior import (
    DefaultPrior,
    UserPriorHardMaxValue,
    UserPriorMaxValue,
)
from botorch.acquisition.user_prior import (
    PiBO,
)

from botorch.acquisition.prior_monte_carlo import (
    qPriorLogExpectedImprovement,
    qPriorMaxValueEntropySearch,
)


def get_test_function(name: str, noise_std: float, seed: int = 0, outputscale=1, task_id=None, bounds=None, fixed_parameters: dict = None):
    
    TEST_FUNCTIONS = {
        'stybtang7': StyblinskiTang(dim=7, noise_std=noise_std, negate=True),
        'levy5': Levy(dim=5, noise_std=noise_std, negate=True),
        'hartmann3': Hartmann(dim=6, noise_std=noise_std, negate=True),
        'hartmann4': Hartmann(dim=4, noise_std=noise_std, negate=True),
        'hartmann6': Hartmann(dim=6, noise_std=noise_std, negate=True),
        'rosenbrock4': Rosenbrock(dim=4, noise_std=noise_std, negate=True),
        'rosenbrock6': Rosenbrock(dim=6, noise_std=noise_std, negate=True),
        'pd1': PD1Function(negate=False, seed=seed, task_id=task_id),
        
    }
    test_function = TEST_FUNCTIONS[name]
    return test_function


ACQUISITION_FUNCTIONS = {
    'MES': qMaxValueEntropy,
    'PiBO': PiBO,
    'MCpi-MES': qPriorMaxValueEntropySearch,
    'MCpi-LogEI': qPriorLogExpectedImprovement,
    'LogNEI': qLogNoisyExpectedImprovement,
}

PRIORS = {
    'default': DefaultPrior,
}

VALUE_PRIORS = {
    'hard': UserPriorHardMaxValue,
    'density': UserPriorMaxValue,
}

INITS = {
    'sobol': Models.SOBOL,
    'prior': Models.PRIOR,
}
