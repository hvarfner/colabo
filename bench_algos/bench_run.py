import sys

import numpy as np
import torch
from torch import Tensor
from botorch.utils.transforms import unnormalize, normalize
from baxus import BAxUS
from baxus.benchmarks import Benchmark
from turbo import Turbo1
from MCTSVS.MCTS import MCTS
from hdbo import algorithms
import cma

method_name =  sys.argv[1]
function_name = sys.argv[2]
seed = int(sys.argv[3])
experiment_name = sys.argv[4]


np.random.seed(seed)
torch.manual_seed(seed)

dt = lambda x: x.detach().numpy()

NEGATE = {
    'baxus': True,
    'turbo': True,
    'cmaes': True,
    'mctsvs': False,
    'rducb': True,
}



from benchmarking.mappings import get_test_function
from benchmarking.external import RecordedTrajectory

NUM_EVALS = 10 * 100
NUM_INIT = 20
test_function = get_test_function(function_name, noise_std=0, seed=seed)
f = RecordedTrajectory(test_function, 
    function_name=function_name, 
    method_name=method_name, 
    experiment_name=experiment_name,
    seed=seed
)
# The world's most convoluted quadruple negation
# Since both benchmarks and methods can require negation
if NEGATE[method_name]:
    f.negate = not f.negate

FUNCTION_WRAP = {
    'cmaes': lambda X: dt(f(Tensor(X))).item(),
}

fun = FUNCTION_WRAP.get(method_name, f)



class MyBenchmark(Benchmark):
    def __init__(self, f):
        lb = dt(f.bounds[0]).flatten()
        ub = dt(f.bounds[1]).flatten()
        dim = f.dim
        super().__init__(lb=lb, ub=ub, dim=dim, noise_std=0)    

    def __call__(self, X):
        return f(Tensor(X))
    
if method_name == 'baxus':
    f_baxus = MyBenchmark(f)
    baxus = BAxUS(
        max_evals=NUM_EVALS,
        n_init=NUM_INIT,
        f=f_baxus,
        target_dim=f.dim,
        verbose=True,
    )
    baxus.optimize()
elif method_name == 'turbo':
    f_turbo = MyBenchmark(f)
    turbo = Turbo1(
        f=f_turbo,
        lb=f_turbo.lb_vec,
        ub=f_turbo.ub_vec,
        n_init=NUM_INIT,
        max_evals = NUM_EVALS,
        batch_size=1,
        verbose=True,
        use_ard=True,
        max_cholesky_size=NUM_EVALS+1,
        n_training_steps=50,
        min_cuda=NUM_EVALS+1,
        device="cpu",
        dtype="float64",
    )
    turbo.optimize()

elif method_name == 'mctsvs':
    f_mcts = MyBenchmark(f)
    mcts = MCTS(
        func=f_mcts,
        dims=f_mcts.dim,
        lb=f_mcts.lb_vec,
        ub=f_mcts.ub_vec,
        Cp=0.1
    )
    mcts.search(max_samples=NUM_EVALS, verbose=False)


elif method_name == 'cmaes':
    sigma_0 = 1
    x0 = dt(unnormalize(torch.rand(f.dim), f.bounds))
    cma.fmin(fun, x0, sigma_0, {'bounds': [dt(f.bounds[0]), dt(f.bounds[1])], 'maxfevals': NUM_EVALS})

    
elif method_name == 'rducb':

    from algorithms import RDUCB
    import init
    from datasets import SyntheticDomain, Function
    init.logger()
    from mlflow_logging import MlflowLogger

    GRID_SIZE = 100
    # we normalize the domain to 0, 1 and scale the input to each benchmark instead
    domain = SyntheticDomain(f.dim, 
        grid_size=GRID_SIZE, 
        domain_lower=0, 
        domain_upper=1
    )
    class RDUCBBenchmark(Function):

        def __init__(self, domain, f):
            Function.__init__(self, domain)
            self.graph = None
            self.f = f
            self.mlflow_logging = MlflowLogger()
        
        def eval(self, X):
            return dt(f(unnormalize(Tensor(X), f.bounds)))
    
    f_rducb = RDUCBBenchmark(domain, fun)

    args = {
    'algorithm': RDUCB,
    'algorithm_random_seed': 2,
    'eps': -1,
    'exploration_weight': lambda t: 0.5 * np.log(2*t),
    'graphSamplingNumIter': 100,
    'initial_kernel_params':
    {
        'lengthscale': 0.1,
        'variance': 0.5,
    },
    'learnDependencyStructureRate': 1,
    'lengthscaleNumIter': 2,
    'max_eval': -4,
    'noise_var': 0.1,
    'param_n_iter': 16,
    'size_of_random_graph': 0.2,
    }
    algorithm = RDUCB(fn=f_rducb, n_iter=NUM_EVALS, n_rand=NUM_INIT, **args)
    algorithm.run()

    # If best regret was reached, no

