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


def get_files_from_experiment(experiment_name, benchmarks=None, acquisitions=None, min_len=-1):
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
            if min_len > 0:
                final_paths = []
                for path in run_paths:
                    with open(path, 'r') as f:
                        test_run = json.load(f)
                        if len(test_run) >= min_len:
                            final_paths.append(path)
            else:
                final_paths = run_paths
            paths_dict[benchmark_name][acq_name] = final_paths

    return paths_dict


def get_hyperparameter_metrics(acquisition_paths, has_outputscale=False, hp_reference=None):
    acq_metrics_per_param = {}
    for acq_name, paths in acquisition_paths.items():
        acq_metrics = {}
        # each path represents one repetition for the given acquisition function
        with open(paths[0], 'r') as f:
            test_run = json.load(f)

        # check the shape of the output so that we can just fill in
        num_iters = len(test_run)
        num_runs = len(paths)
        first_iter = test_run['iter_10']
        try:
            num_hp_sets = len(first_iter[list(first_iter.keys())[0]])
        except:
            print('Forcing HP sets to 1')
            num_hp_sets = 1

        all_hp_names = []
        for hp_name, values in first_iter.items():
            array_vals = np.array(values)
            if array_vals.ndim == 1 or array_vals.shape[-1] == 1:
                all_hp_names.append(hp_name)
            else:
                for dim in range(array_vals.shape[-1]):
                    all_hp_names.append(hp_name + f'_{dim+1}')

        for hp_name in all_hp_names:
            acq_metrics[hp_name] = np.zeros((num_iters, num_runs, num_hp_sets))

        for run_idx, path in enumerate(paths):
            # print(run_idx)
            # we read in the path and retrieve the hyperparameters per iteration
            with open(path, 'r') as f:
                run_hyperparamers = json.load(f)
                for iteration in range(7, len(run_hyperparamers)):
                    # TODO fix this is we use outputscale as well
                    # TODO check the shape of the outputscale first
                    if iteration == 0:
                        included_hps = list(run_hyperparamers[f'iter_10'].keys())

                    hps = run_hyperparamers[f'iter_{iteration}']

                    for hp_name, values in hps.items():
                        array_vals = np.array(values)
                        if array_vals.ndim == 1 or array_vals.shape[-1] == 1:
                            dim_name = hp_name
                            acq_metrics[dim_name][iteration,
                                                  run_idx, :] = array_vals.flatten()

                        else:
                            for dim in range(array_vals.shape[-1]):
                                dim_name = f'{hp_name}_{dim+1}'
                                acq_metrics[dim_name][iteration, run_idx,
                                                      :] = array_vals[..., dim].flatten()

                    # TODO here's where we add a reference if we have one
        acq_metrics_per_param[acq_name] = acq_metrics

    return acq_metrics_per_param
    # compute the mean (log) hyperparameter value per dim?


def read_benchmark_hps(benchmark, reference):
    path_to_hps = join(dirname(dirname(abspath(__file__))),
                       f'results/{reference}/{benchmark}/Dummy/{benchmark}_Dummy_run_42_hps.json')
    with open(path_to_hps, 'r') as f:
        hps = json.load(f)

    # all iterations have the same HPs, anyway
    hps = hps[list(hps.keys())[0]]
    outputscales = np.array(hps['outputscale']).reshape(-1, 1)
    noises = np.array(hps['noise'])
    lengthscales = np.array(hps['lengthscales']).squeeze(1)
    hp_array = np.append(outputscales, noises, axis=1)
    hp_array = np.append(hp_array, lengthscales, axis=1)
    hp_log_array = np.log10(hp_array)
    return hp_log_array


def get_hp_order(hps):
    sorted_hps = []
    concentrations0 = []
    concentrations1 = []
    lengthscales = []
    for hp in hps.keys():
        if 'noise' in hp:
            sorted_hps.append(hp)
        elif 'outputscale' in hp:
            sorted_hps.append(hp)
        elif 'lengthscale' in hp:
            lengthscales.append(hp)
        elif 'concentration0' in hp:
            concentrations0.append(hp)
        elif 'concentration1' in hp:
            concentrations1.append(hp)
        else:
            raise ValueError(f'HP {hp} is not available for sorting.')

    def sort_func(name): return int(name.split('_')[-1])
    lengthscales = sorted(lengthscales, key=sort_func)
    concentrations0 = sorted(concentrations0, key=sort_func)
    concentrations1 = sorted(concentrations1, key=sort_func)
    sorted_hps.extend(lengthscales)
    sorted_hps.extend(concentrations0)
    sorted_hps.extend(concentrations1)
    return sorted_hps


def plot_per_param(experiment, benchmark, acquisitions, reference, hp_type=0, only_lengthscales=False, drop_outputscale=False, num_hps=4, min_len=-1):
    all_files = get_files_from_experiment(
        experiment, benchmark, acquisitions, min_len=min_len)
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    MAX = 7

    acq_per_param = get_hyperparameter_metrics(all_files[benchmark])
    plots_per_row = num_hps

    if drop_outputscale:
        for key in acq_per_param.keys():
            acq_per_param[key].pop('outputscale')
    num_params = len(acq_per_param[list(acq_per_param.keys())[0]])
    if MAX > 0:
        num_params = MAX
    fig, axes = plt.subplots(
        np.ceil((num_params - 1) / plots_per_row).astype(int), plots_per_row, figsize=(20, 3.2))
    axes[0].set_ylabel('Log hyperparam. value', fontsize=16)
    for acq, hps in acq_per_param.items():
        try:
            hps.pop('mean')
            # hps.pop('noise')

        except:
            print(f'Acq. {acq} has no mean HP.')
        hp_order = get_hp_order(hps)
        for hp_idx, hp_name in enumerate(hp_order):
            if hp_idx == MAX:
                break
            values = hps[hp_name]
            row = int(hp_idx / plots_per_row)
            col = hp_idx % plots_per_row
            if axes.ndim == 1:
                ax_ = axes[col]

            else:
                ax_ = axes[row, col]
            ax_.set_title(get_hp_name(hp_name), fontsize=26)
            try:
                ax_.axhline(reference[get_hp_name(hp_name)],
                            color='k', linestyle='dotted', linewidth=2)
            except:
                pass
            plot_layout = copy(PLOT_LAYOUT)
            plot_layout['c'] = COLORS.get(acq, 'k')

            plot_layout['markevery'] = 500
            plot_layout['linewidth'] = 1
            x = np.arange(1, len(values) + 1)
            # print(vals.mean(axis=1))
            vals = values.reshape(len(values), -1)
            if hp_name != 'mean':
                vals = np.log10(vals)
            mean = vals.mean(axis=1)
            MA = 5
            mean = np.convolve(mean, np.ones(MA), 'valid') / MA
            err = np.convolve(np.std(vals, axis=1), np.ones(MA), 'valid') / MA

            # for i, color in enumerate(colors):
            #
            #    plot_layout['c'] = color
            #    ax_.plot(x, vals[:, i, :], label=acq, **plot_layout)
            ax_.plot(x[(MA - 1):], mean, label=NAMES[acq], **plot_layout)
            ax_.fill_between(x[(MA - 1):], mean - err, mean + err,
                             alpha=0.2, color=plot_layout['c'])
            ax_.plot(x[(MA - 1):], mean + err, alpha=0.05, color=plot_layout['c'])
            ax_.plot(x[(MA - 1):], mean - err, alpha=0.05, color=plot_layout['c'])
            try:
                ax_.axhline(reference[hp_type])
            except:
                pass
        # ax.axvline(x=init, color='k', linestyle=':', linewidth=4)
            if row == np.ceil((num_params - 1) / plots_per_row) - 1:
                ax_.tick_params(axis='x', labelsize=15)
            else:
                ax_.get_xaxis().set_visible(False)

            ax_.tick_params(axis='y', labelsize=15)
            ax_.locator_params(axis='y', nbins=4)
            ax_.grid(visible=True)

    ax_.legend(fontsize=18)

    plt.subplots_adjust(left=0.04,
                        bottom=0.09,
                        right=0.99,
                        top=0.86,
                        wspace=0.14,
                        hspace=0.0)
    plt.savefig('neurips_plots/saasbo_new.pdf')
    plt.show()


if __name__ == '__main__':
    #acqs = ['BQBC', 'QBMGP', 'SAL_HR', 'SAL_WS', 'BALM']
    acqs = [
        'JES-e-LB2',
        'GIBBON',
        'NEI',
        'ScoreBO_J_HR'
    ]
    # acqs = [
    #    'BOTorch_mean_prior',
    #    'ALBO_prior'
    # ]
    # acqs = 'ScoreBO_J_HR', 'NEI', 'ScoreBO_M_HR', 'ScoreBO_large'
    hp_reference = {
        '$\sigma^2$': np.log10(10**2.2),
        '$\sigma_\epsilon^2$': np.log10(0.187**2),  # np.log10(0.0192**2),
        '$\ell_1$': np.log10(0.35),  # np.log(2),  # np.log10(0.35), #np.log10(0.45),
        # np.log(4),  # np.log10(0.27), #np.log10(0.27 + 0.12)
        '$\ell_2$': np.log10(0.27),
        # np.log(2),  # np.log10(0.33), #np.log10(0.33 + 0.20)
        '$\ell_3$': np.log10(0.33),
        '$\ell_4$': None,  # np.log(0.4),
        '$\ell_5$': None,  # np.log(0.5),
        '$\ell_6$': None,  # np.log(0.4),
        '$\ell_7$': None,
        '$\ell_8$': None,
        '$c_0, 1$': 0.0,
        '$c_0, 2$': 0.0,
        '$c_0, 3$': 0.0,
        '$c_0, 4$': 0.0,
        '$c_1, 1$': 0.0,
        '$c_1, 2$': 0.0,
        '$c_1, 3$': 0.0,
        '$c_1, 4$': 0.0,
    }
    hp_reference = {
        '$\sigma^2$': None,
        '$\sigma_\epsilon^2$': -1,  # np.log10(0.0192**2),
        '$\ell_1$': -1,  # np.log(2),  # np.log10(0.35), #np.log10(0.45),
        # np.log(4),  # np.log10(0.27), #np.log10(0.27 + 0.12)
        '$\ell_2$': -1,
        # np.log(2),  # np.log10(0.33), #np.log10(0.33 + 0.20)
        '$\ell_3$': -1,
        '$\ell_4$': -1,  # np.log(0.4),
        '$\ell_5$': None,  # np.log(0.5),
        '$\ell_6$': None,  # np.log(0.4),
        '$\ell_7$': None,
        '$\ell_8$': None,
        '$c_0, 1$': 0.0,
        '$c_0, 2$': 0.0,
        '$c_0, 3$': 0.0,
        '$c_0, 4$': 0.0,
        '$c_1, 1$': 0.0,
        '$c_1, 2$': 0.0,
        '$c_1, 3$': 0.0,
        '$c_1, 4$': 0.0,
    }
    # hp_reference = None
    plot_per_param(
        'results/20230524_ScoreBO_Brutal_noise', 'levy5',
        acqs,
        reference=hp_reference,
        only_lengthscales=True,
        drop_outputscale=False,
        num_hps=7,
        min_len=190
    )
