import math

from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from omegaconf import DictConfig
from benchmarking.prior_utils import sample_offset, sample_random
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    _GaussianLikelihoodBase,
    FixedNoiseGaussianLikelihood,
    GaussianLikelihood,
)
from botorch.utils.transforms import normalize
from operator import attrgetter
from gpytorch.priors import GammaPrior
from benchmarking.eval_utils import (
    get_model_hyperparameters
)
from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS,
    PRIORS,
    VALUE_PRIORS,
    INITS
)
from benchmarking.mapping.gp_priors import (
    MODELS,
    get_covar_module

)
import numpy as np
from torch.quasirandom import SobolEngine
from benchmarking.eval_utils import compute_ws_and_acq
import os
from os.path import dirname, abspath, join
import sys
import json
import hydra
import torch
from torch import Tensor
from time import time
sys.path.append('.')


MIN_INFERRED_NOISE_LEVEL = 1e-4
N_VALID_SAMPLES = int(250)


@hydra.main(config_path='./../configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    print(cfg)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    BETA_DEFAULT = cfg.prior.get('default_beta', 1.0)

    benchmark = cfg.benchmark.name
    q = int(cfg.q)
    if hasattr(cfg.benchmark, 'outputscale'):
        test_function = get_test_function(
            benchmark, float(cfg.benchmark.noise_std), cfg.seed, float(cfg.benchmark.outputscale))

    else:
        test_function = get_test_function(
            name=cfg.benchmark.benchmark,
            noise_std=float(cfg.benchmark.noise_std),
            seed=cfg.seed,
            task_id=cfg.benchmark.get('task_id', None),
            fixed_parameters=cfg.benchmark.get('fixed_parameters'))

    num_init = max(cfg.benchmark.num_init, cfg.q)
    num_bo = cfg.benchmark.num_iters - num_init
    if cfg.algorithm.name == 'Sampling':
        num_init = cfg.benchmark.num_iters
        num_bo = 0
    acq_func = ACQUISITION_FUNCTIONS[cfg.algorithm.acq_func]
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0)

    if hasattr(cfg.algorithm, 'acq_kwargs'):
        acq_func_kwargs = dict(cfg.algorithm.acq_kwargs)
    else:
        acq_func_kwargs = {}

    if cfg.prior is not None and cfg.algorithm.get('use_prior', False):
        prior_type = PRIORS[cfg.prior.type]
        location = cfg.prior.parameters.location
        confidence = cfg.prior.parameters.confidence
        offset = cfg.prior.parameters.offset

        if location is None:
            if 'gp' in benchmark:
                root = dirname(abspath(__file__))
                path = f'{root}/gp_sample/{benchmark}/{benchmark}_run{cfg.seed}.json'
                with open(path, 'r') as f:
                    location = Tensor(json.load(f)['X']).flatten()
            else:
                location = cfg.benchmark.prior_location
                location = Tensor(location)

        if offset == -1:
            location = sample_random(bounds)

        else:
            location = sample_offset(location, bounds, offset)

        user_prior = prior_type(bounds, location, confidence=confidence)
        acq_func_kwargs['user_prior'] = user_prior
        if not acq_func_kwargs.get('decay_beta', False):
            acq_func_kwargs['decay_beta'] = cfg.benchmark.num_iters / BETA_DEFAULT

    else:
        user_prior = None

    if cfg.model.model_kwargs is None:
        prior_location = None if user_prior is None else normalize(
            location, Tensor(cfg.benchmark.bounds).T)

        model_kwargs = get_covar_module(cfg.model.model_name, len(
            bounds.T), 
            prior_location=prior_location, 
            gp_params=cfg.model.get('gp_params', None),
            gp_constraints=cfg.model.get('gp_constraints', {})
        )
    else:
        model_kwargs = dict(cfg.model.model_kwargs)

    refit_on_update = not hasattr(cfg.model, 'model_parameters')

    refit_params = {}

    if not refit_on_update:
        model_enum = Models.BOTORCH_MODULAR_NOTRANS
        if hasattr(cfg.benchmark, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)

        # if the model also has fixed parameters, we will override with those
        if hasattr(cfg.model, 'model_parameters'):
            params = cfg.benchmark.model_parameters
            refit_params['outputscale'] = Tensor(params.outputscale)
            refit_params['lengthscale'] = Tensor(params.lengthscale)

    else:
        model_enum = Models.BOTORCH_MODULAR

    if cfg.algorithm.get('use_prior', False) and cfg.algorithm.init_type == 'prior':
        init_type = INITS['prior']
        init_type_2 = INITS['prior']
        init_kwargs = {"seed": int(cfg.seed), "prior": user_prior}
        other_init_kwargs = {**init_kwargs, 'generate_default': False}

    elif cfg.algorithm.get('use_prior', False) and cfg.algorithm.init_type == 'sobol':
        init_type = INITS['prior']
        init_type_2 = INITS['sobol']

        init_kwargs = {"seed": int(cfg.seed), "prior": user_prior}
        other_init_kwargs = {"seed": int(cfg.seed), }

    else:
        init_type = INITS['sobol']

    if cfg.get('prior_value', False) and cfg.algorithm.use_prior:
        prior_value_type = VALUE_PRIORS[cfg.prior_value.type]
        if cfg.prior_value.type == 'density':
            if cfg.prior_value.parameters.value is None:
                value = cfg.benchmark.prior_value
            else:
                value = cfg.prior_value.parameters.value

            if cfg.prior_value.parameters.confidence is None:
                confidence = cfg.benchmark.prior_confidence
            else:
                confidence = cfg.prior_value.parameters.confidence

            user_prior_value = prior_value_type(
                parameter_default=value, confidence=confidence)

        elif cfg.prior_value.type == 'hard':
            minopt_value = cfg.prior_value.parameters.minopt_value
            if minopt_value is None:
                minopt_value = cfg.benchmark.get('minopt_value', None)
            maxopt_value = cfg.prior_value.parameters.maxopt_value
            if maxopt_value is None:
                maxopt_value = cfg.benchmark.get('maxopt_value', None)

            user_prior_value = prior_value_type(
                minopt_value=minopt_value, maxopt_value=maxopt_value)
        acq_func_kwargs['user_prior_value'] = user_prior_value

    if init_type is Models.PRIOR:
        steps = [
            GenerationStep(
                model=init_type,
                num_trials=1,
                # First, we generate the default config (prior mode)
                model_kwargs={**init_kwargs},
            ),
        ]
        if num_init > 1:
            steps.append(GenerationStep(
                model=init_type_2,
                # How many trials should be produced from this generation step
                num_trials=num_init - 1,
                # Then, samples from the prior for the remainder
                model_kwargs={**other_init_kwargs},
            ))

    else:
        init_kwargs = {"seed": int(cfg.seed)}
        steps = [
            GenerationStep(
                model=init_type,
                num_trials=num_init,
                # Otherwise, it's probably just SOBOL
                model_kwargs=init_kwargs,
            )]

    opt_setup = cfg.acq_opt
    model = MODELS[cfg.model.gp]
    min_noise_level = None
    if cfg.algorithm.get('acq_kwargs', False):
        min_noise_level = cfg.algorithm.acq_kwargs.get('min_noise', None)

    if min_noise_level is not None:
        noise_prior = GammaPrior(0.001, 0.01)
        noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate

        likelihood = GaussianLikelihood(
            noise_prior=noise_prior,
            noise_constraint=GreaterThan(
                min_noise_level,
                transform=None,
                initial_value=noise_prior_mode,
            ),
        )
        model_kwargs['likelihood'] = likelihood

    if model_kwargs.get('train_Yvar', None) is not None:
        model_kwargs['train_Yvar'] = torch.Tensor(
            [model_kwargs['train_Yvar']]).to(torch.float64)

    if num_bo > 0:
        bo_step = GenerationStep(
            # model=model_enum,
            model=model_enum,
            # No limit on how many generator runs will be produced
            num_trials=num_bo,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate": Surrogate(
                            botorch_model_class=model,
                            model_options=model_kwargs
                ),
                "botorch_acqf_class": acq_func,
                "acquisition_options": {**acq_func_kwargs},
                "refit_on_update": refit_on_update,
                "refit_params": refit_params
            },
            model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
                "optimizer_kwargs": dict(opt_setup)},
            },
        )
    if cfg.algorithm.name != 'Sampling' and num_bo > 0:
        steps.append(bo_step)

    def evaluate(parameters, seed=None):
        x = torch.tensor(
            [[parameters.get(f"x_{i+1}") for i in range(test_function.dim)]])

        if seed is not None:
            bc_eval = test_function.evaluate_true(x, seed=seed).squeeze().tolist()
        else:
            bc_eval = test_function(x).squeeze().tolist()

        return {benchmark: bc_eval}

    gs = GenerationStrategy(
        steps=steps
    )

    # Initialize the client - AxClient offers a convenient API to control the experiment
    ax_client = AxClient(generation_strategy=gs)
    # Setup the experiment
    ax_client.create_experiment(
        name=cfg.experiment_name,
        parameters=[
            {
                "name": f"x_{i+1}",
                "type": "range",
                # It is crucial to use floats for the bounds, i.e., 0.0 rather than 0.
                # Otherwise, the parameter would
                "bounds": bounds[:, i].tolist(),
            }
            for i in range(test_function.dim)
        ],
        objectives={
            benchmark: ObjectiveProperties(minimize=False),
        },
    )
    best_guesses = [0] * num_init
    hyperparameters = {}
    wass_arr = [0] * num_init
    acqdiff_arr = [0] * num_init
    wass_arr_local = [0] * num_init
    acqdiff_arr_local = [0] * num_init
    rmse_arr = [0] * num_init
    true_vals = []
    guess_vals = [0.0] * (num_init)

    best_guesses = []
    hyperparameters = {}
    likelihoods = {}
    likelihoods['other_likelihoods'] = []
    scale_hyperparameters = model_enum != Models.BOTORCH_MODULAR_NOTRANS
    bo_times = []

    total_iters = num_init + num_bo
    total_batches = math.ceil((num_init + num_bo) / q)
    current_count = 0

    for i in range(total_batches):
        current_count = (q * i)
        batch_data = []
        q_curr = min(q, total_iters - current_count)
        if current_count >= num_init:
            start_time = time()

        for q_rep in range(q_curr):
            batch_data.append(ax_client.get_next_trial())
        if current_count >= num_init:
            end_time = time()
            bo_times.append(end_time - start_time)
        # Local evaluation here can be replaced with deployment to external system.
        for q_rep in range(q_curr):
            parameters, trial_index = batch_data[q_rep]
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=evaluate(parameters))
            # Unique sobol draws per seed and iteration, but identical for different acqs
            if current_count < num_init:
                best_guesses.append(parameters)

        results_df = ax_client.get_trials_data_frame()
        configs = torch.tensor(
            results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())

        if cfg.benchmark.get('synthetic', True):
            for q_idx in range(q_curr):
                true_vals.append(test_function.evaluate_true(
                    configs[-q_curr + q_idx].unsqueeze(0)).item())
            results_df['True Eval'] = true_vals
            infer_values = None

        if current_count >= num_init and q == 1:

            model = ax_client._generation_strategy.model.model.surrogate.model

            sobol = SobolEngine(dimension=test_function.dim,
                                scramble=True, seed=i + cfg.seed * 100)
            # crucially, these are already in [0, 1]
            test_samples = sobol.draw(n=N_VALID_SAMPLES)

            acq = ax_client._generation_strategy.model.model.evaluate_acquisition_function
            current_data = ax_client.get_trials_data_frame()[benchmark].to_numpy()
            hps = get_model_hyperparameters(
                model, current_data, scale_hyperparameters=scale_hyperparameters, objective=test_function, acquisition=acq)
            hyperparameters[f'iter_{i}'] = hps
            if 'MCpi' in cfg.algorithm.name and cfg.algorithm.get('test_gp', False):

                wass, acq_diff = compute_ws_and_acq(
                    ax_client, test_samples, test_function, benchmark)
                wass_local, acq_diff_local = compute_ws_and_acq(
                    ax_client, test_samples, test_function, benchmark, local=True)

                for q_rep in range(q_curr):
                    wass_arr.append(wass.mean().item())
                    acqdiff_arr.append(acq_diff.item())
                    wass_arr_local.append(wass_local.mean().item())
                    acqdiff_arr_local.append(acq_diff_local.item())

            else:
                observation_guesses = model.posterior(model.train_inputs[0]).mean
                from botorch.utils.transforms import unnormalize
                best_obs = unnormalize(
                    model.train_inputs[0][torch.argmax(observation_guesses)], bounds)
                val = test_function.evaluate_true(best_obs.unsqueeze(0))
                guess_vals.append(val.item())
                for q_rep in range(q_curr):
                    results_df['Guess values'] = guess_vals


        mod_result_path = cfg.result_path
        os.makedirs(mod_result_path, exist_ok=True)

        with open(f"{mod_result_path}/{ax_client.experiment.name}_hps.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)
        results_df.to_csv(f"{mod_result_path}/{ax_client.experiment.name}.csv")


if __name__ == '__main__':
    main()
