import sys
import os
from os.path import join, dirname, isdir, abspath
from glob import glob

import json
from copy import copy
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
from scipy.stats import sem
import pandas as pd
from scipy.linalg import norm as euclidian_norm
from pandas.errors import EmptyDataError
from benchmarking.scoreboplot import COLORS, init, NAMES, PLOT_LAYOUT


plt.rcParams['font.family'] = 'serif'
NUM_MARKERS = 10
colors = [
    '#377eb8', '#ff7f00', '#4daf4a',
    '#f781bf'
]


def get_hp_name(name):
    if 'outputscale' in name:
        return '$\sigma^2$'
    if 'noise' in name:
        return '$\sigma_\epsilon^2$'
    if 'length' in name:
        dim = name.split('_')[-1]
        return f'$\ell_{dim}$'
    if 'concentration' in name:
        c_name, dim = name.split('_')
        c_name = c_name[-1]
        return f'$c_{c_name}, {dim}$'

    return name


def filter_paths(all_paths, included_names=None):
    all_names = [benchmark_path.split('/')[-1]
                 for benchmark_path in all_paths]
    if included_names is not None:
        used_paths = []
        used_names = []

        for path, name in zip(all_paths, all_names):
            if name in included_names:
                used_paths.append(path)
                used_names.append(name)
        return used_paths, used_names

    return all_paths, all_names


def get_files_from_experiment(experiment_name, benchmarks=None, acquisitions=None):
    '''
    For a specific expefiment, gets a dictionary of all the {benchmark: {method: [output_file_paths]}}
    as a dict, includiong all benchmarks and acquisition functions unless specified otherwise in 
    the arguments.
    '''
    paths_dict = {}
    all_benchmark_paths = glob(join(experiment_name, '*'))
    filtered_benchmark_paths, filtered_benchmark_names = filter_paths(
        all_benchmark_paths, benchmarks)

    # *ensures hidden files are not included
    for benchmark_path, benchmark_name in zip(filtered_benchmark_paths, filtered_benchmark_names):
        paths_dict[benchmark_name] = {}
        all_acq_paths = glob(join(benchmark_path, '*'))
        filtered_acq_paths, filtered_acq_names = filter_paths(
            all_acq_paths, acquisitions)

        for acq_path, acq_name in zip(filtered_acq_paths, filtered_acq_names):
            run_paths = glob(join(acq_path, '*.json'))
            paths_dict[benchmark_name][acq_name] = run_paths

    return paths_dict


def get_likelihood_from_experiment(experiment_name, benchmarks=None, acquisitions=None):
    '''
    For a specific expefiment, gets a dictionary of all the {benchmark: {method: [output_file_paths]}}
    as a dict, includiong all benchmarks and acquisition functions unless specified otherwise in 
    the arguments.
    '''
    paths_dict = {}
    all_benchmark_paths = glob(join(experiment_name, '*'))
    filtered_benchmark_paths, filtered_benchmark_names = filter_paths(
        all_benchmark_paths, benchmarks)

    # *ensures hidden files are not included
    for benchmark_path, benchmark_name in zip(filtered_benchmark_paths, filtered_benchmark_names):
        paths_dict[benchmark_name] = {}
        all_acq_paths = glob(join(benchmark_path, '*'))
        filtered_acq_paths, filtered_acq_names = filter_paths(
            all_acq_paths, acquisitions)

        for acq_path, acq_name in zip(filtered_acq_paths, filtered_acq_names):
            run_paths = glob(join(acq_path, '*.json'))
            paths_dict[benchmark_name][acq_name] = run_paths

    return paths_dict


def get_dims(benchmark, files, dim_reference):
    acqs = {}
    dim_reference = np.array(dim_reference)
    from ast import literal_eval
    benchmark_files = files[benchmark]
    for key, runs in benchmark_files.items():
        acqs[key] = None
        for run_idx, path in enumerate(runs):
            acqs[key]
            with open(path, 'r') as f:
                lik_df = pd.read_csv(f)
                if acqs[key] is None:
                    acqs[key] = np.zeros((len(lik_df), 0))
                lik = lik_df['bis_likelihood'].to_numpy().reshape(-1, 1)
                other_lik = np.array([literal_eval(lik)
                                      for lik in lik_df['other_likelihoods']])
                lik_diff = lik - other_lik
                acqs[key] = np.append(acqs[key], lik_diff, axis=1)
    return acqs


def get_groups(benchmark, files, dim_reference):
    acqs = {}
    dim_reference = np.array(dim_reference)
    benchmark_files = files[benchmark]
    for key, runs in benchmark_files.items():
        run_list = []
        for run_idx, path in enumerate(runs):

            run_metrics = []
            with open(path, 'r') as f:
                run_hyperparamers = json.load(f)

            for iteration in range(5, len(run_hyperparamers)):
                # TODO fix this is we use outputscale as well
                # TODO check the shape of the outputscale first
                if iteration == 0:
                    included_hps = list(run_hyperparamers[f'iter_10'].keys())
                    acqs[key] = []
                groups = np.array(
                    run_hyperparamers[f'iter_{iteration}']['active_groups'])

                distance = -len(dim_reference)
                for ref_group in dim_reference:
                    relevant_dims = groups[..., ref_group]

                    batches_to_count = (relevant_dims.sum(axis=-1) > 0).sum()
                    # count the number of splits in each group (is the group not split, split in 2, 3 etc.)
                    distance += batches_to_count / groups.shape[1]

                run_metrics.append(distance)
            run_list.append(run_metrics)

        acqs[key] = np.array(run_list)
    return acqs


def plot_per_param(experiment, benchmark, acquisitions, dim_reference, hp_type=0, only_lengthscales=False, maxlen=-1):
    all_files = get_likelihood_from_experiment(experiment, benchmark, acquisitions)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    dim_classes = get_groups(benchmark, all_files, dim_reference)
    plots_per_row = 1
    num_params = 1
    fig, axes = plt.subplots(
        np.ceil((num_params) / plots_per_row).astype(int), plots_per_row, figsize=(5, 3.8))

    for acq, values in dim_classes.items():
        ax_ = axes
        ax_.set_title('Incorrectly split components', fontsize=22)
        try:
            ax_.axhline(reference[get_hp_name(hp_name)],
                        color='k', linestyle='dotted', linewidth=2)
        except:
            pass
        plot_layout = copy(PLOT_LAYOUT)
        plot_layout['c'] = COLORS.get(acq, 'k')

        plot_layout['markevery'] = 500
        plot_layout['linewidth'] = 1

        # print(vals.mean(axis=1))
        if maxlen > 0:
            # .reshape(len(values), -1)
            vals = np.array([v[0:maxlen] for v in values]).T
        else:
            vals = np.array([v for v in values]).T

        x = np.arange(1, len(vals) + 1)
        MA = 5

        mean = vals.mean(axis=-1)
        mean = np.convolve(mean, np.ones(MA), 'valid') / MA
        err = np.std(vals, axis=-1) * 0.5
        err = np.convolve(np.std(vals, axis=1), np.ones(MA), 'valid') / MA

        ax_.plot(x[(MA - 1):], mean, label=NAMES[acq], **plot_layout)
        ax_.fill_between(x[(MA - 1):], mean - err, mean + err,
                         alpha=0.3, color=plot_layout['c'])
        ax_.plot(x[(MA - 1):], mean - err, alpha=0.1, color=plot_layout['c'])
        ax_.plot(x[(MA - 1):], mean + err, alpha=0.1, color=plot_layout['c'])
    # ax_.axhline(refs[hp_type])

    #ax.axvline(x=init, color='k', linestyle=':', linewidth=4)

        ax_.tick_params(axis='y', labelsize=15)
        ax_.tick_params(axis='x', labelsize=15)
        ax_.locator_params(axis='y', nbins=4)

    plt.legend(fontsize=19)
    # plt.tight_layout()

    #ax_.set_xlabel('Iteration', fontsize=16)
    plt.savefig('neurips_plots/addgp_hp.pdf')
    plt.show()


if __name__ == '__main__':
    task = 'gp_2_2_2dim'
    acqs = ['ScoreBO_J_HR', 'NEI']
    # acqs = [
    #    'ScoreBO_J_HR',
    #    'ScoreBO_J_HR_MMfix',
    #    'ScoreBO_J_HR_MC64_v3',
#
    # ]
    # acqs = [
    #    'BOTorch_mean_prior',
    #    'ALBO_prior'
    # ]
    #acqs = 'ScoreBO_J_HR', 'NEI', 'ScoreBO_M_HR', 'ScoreBO_large'
    hp_reference = {
        '$\sigma^2$': 0,
        '$\sigma_\epsilon^2$': -1,
        '$\ell_1$': -1,
        '$\ell_2$': -0.5,
        '$\ell_3$': -0.5,
        '$\ell_4$': 0,
        '$\ell_5$': 0,
        '$\ell_6$': 1.5,
        '$\ell_7$': 1.5,
        '$\ell_8$': 1.5,
        '$c_0, 1$': 0.0,
        '$c_0, 2$': 0.0,
        '$c_0, 3$': 0.0,
        '$c_0, 4$': 0.0,
        '$c_1, 1$': 0.0,
        '$c_1, 2$': 0.0,
        '$c_1, 3$': 0.0,
        '$c_1, 4$': 0.0,
    }
    dim_reference = {
        'botorch_3_3_2dim': [
            [True, True, True, False, False, False, False, False],
            [False, False, False, True, True, True, False, False],
            [False, False, False, False, False, False, True, True],
        ],
        'gp_2_2_2_2dim': [
            [True, True, False, False, False, False, False, False],
            [False, False, True, True, False, False, False, False],
            [False, False, False, False, True, True, False, False],
            [False, False, False, False, False, False, True, True],
        ],
        'gp_2_2_2_2_2_2dim': [
            [True, True, False, False, False, False,
                False, False, False, False, False, False],
            [False, False, True, True, False, False,
                False, False, False, False, False, False],
            [False, False, False, False, True, True,
                False, False, False, False, False, False],
            [False, False, False, False, False, False,
                True, True, False, False, False, False],
            [False, False, False, False, False, False,
                False, False, True, True, False, False],
            [False, False, False, False, False, False,
                False, False, False, False, True, True],
        ],
        'gp_2_2_2_2_2dim': [
            [True, True, False, False, False, False, False, False, False, False],
            [False, False, True, True, False, False, False, False, False, False],
            [False, False, False, False, True, True, False, False, False, False],
            [False, False, False, False, False, False, True, True, False, False],
            [False, False, False, False, False, False, False, False, True, True],
        ],
        'gp_2_2_2dim': [
            [True, True, False, False, False, False],
            [False, False, True, True, False, False],
            [False, False, False, False, True, True],
        ],
        'botorch_3_3_4dim': [
            [True, True, True, False, False, False, False, False, False, False],
            [False, False, False, True, True, True, False, False, False, False],
            [False, False, False, False, False, False, True, True, True, True],
        ],
        'gp_2_2dim': [
            [True, True, False, False],
            [False, False, True, True],
        ],
    }
    maxlen = {
        'gp_2_2_2dim': 100
    }
    plot_per_param(
        'results/20230420_addGP_rbf05',
        task,
        acqs,
        dim_reference=dim_reference[task],
        only_lengthscales=True,
        maxlen=maxlen.get(task, -1)
    )
