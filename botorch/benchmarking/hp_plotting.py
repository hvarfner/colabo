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
from benchmarking.plot import COLORS, init, NAMES, PLOT_LAYOUT


plt.rcParams['font.family'] = 'serif'
NUM_MARKERS = 10


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


def get_hyperparameter_metrics(acquisition_paths, has_outputscale=False, hp_reference=None):
    acq_metrics = {}
    for acq_name, paths in acquisition_paths.items():
        # each path represents one repetition for the given acquisition function
        with open(paths[0], 'r') as f:
            test_run = json.load(f)

        # check the shape of the output so that we can just fill in
        divergences = np.zeros((len(test_run), len(paths)))

        for run_idx, path in enumerate(paths):
            # print(run_idx)
            # we read in the path and retrieve the hyperparameters per iteration
            with open(path, 'r') as f:
                run_hyperparamers = json.load(f)
                for iteration in range(len(run_hyperparamers)):
                    # TODO fix this is we use outputscale as well
                    # TODO check the shape of the outputscale first
                    if iteration == 0:
                        included_hps = list(run_hyperparamers[f'iter_0'].keys())

                    hps = run_hyperparamers[f'iter_{iteration}']
                    outputscales = np.array(hps['outputscale']).reshape(-1, 1)
                    noises = np.array(hps['noise'])
                    lengthscales = np.array(hps['lengthscales']).squeeze(1)
                    hp_array = np.append(lengthscales, noises, axis=1)
                    hp_array = np.append(hp_array, outputscales, axis=1)

                    hp_log_array = np.log10(hp_array)
                    # TODO here's where we add a reference if we have one
                    mean_hp_set = hp_log_array.mean(axis=0)
                    if hp_reference is None:
                        distance_to_mean = euclidian_norm(
                            mean_hp_set - hp_log_array, axis=1)
                        divergences[iteration, run_idx] = distance_to_mean.mean()
                    else:
                        distance_to_mean = euclidian_norm(
                            hp_reference - hp_log_array, axis=1)
                        #print(acq_name, distance_to_mean)
                        divergences[iteration, run_idx] = distance_to_mean.mean()
                print(acq_name, (hp_log_array - hp_reference).mean(axis=0))
        acq_metrics[acq_name] = divergences

    return acq_metrics
    # compute the mean (log) hyperparameter value per dim?


def read_hyperparameter_refs(benchmark):
    path_to_hps = join(dirname(abspath(__file__)),
                       f'hyperparameters/{benchmark}_hps.json')
    with open(path_to_hps, 'r') as f:
        hps = json.load(f)
    outputscales = np.array(hps['outputscale']).reshape(-1, 1)
    noises = np.array(hps['noise']).reshape(-1, 1)
    lengthscales = np.array(hps['lengthscales']).reshape(-1, 1)
    hp_array = np.append(lengthscales, noises, axis=0)
    hp_array = np.append(hp_array, outputscales, axis=0)

    hp_log_array = np.log10(hp_array)
    return hp_log_array.T

def plot_hyperparameters(experiment, benchmarks, acquisitions):
    all_files = get_files_from_experiment(experiment, benchmarks, acquisitions)
    if len(benchmarks) > 3:
        num_rows = 2
    else:
        num_rows = 1
    cols = int(len(benchmarks) / num_rows)
    fig, axes = plt.subplots(num_rows, cols, figsize=(25, 9))
    for benchmark_idx, (benchmark_name, paths) in enumerate(all_files.items()):
        hp_ref = read_hyperparameter_refs(benchmark_name)
        
        acq_metrics = get_hyperparameter_metrics(paths, hp_reference=hp_ref)
        if num_rows == 1:
            ax = axes[benchmark_idx]
        else:
            ax = axes[int(benchmark_idx / cols), benchmark_idx % cols]

        ax.set_title(benchmark_name, fontsize=24)
        for acq, values in acq_metrics.items():
            print('Benchmark:', benchmark_name, '     Acq:', acq)
            plot_layout = copy(PLOT_LAYOUT)
            plot_layout['c'] = COLORS.get(acq, 'k')

            markevery = np.floor(len(values) / NUM_MARKERS).astype(int)
            plot_layout['markevery'] = markevery

            x = np.arange(1, len(values) + 1)
            mean = values.mean(axis=1)
            err = sem(values, axis=1)
            ax.plot(x, mean, label=acq, **plot_layout)
            ax.fill_between(x, mean - err, mean + err,
                            alpha=0.2, color=plot_layout['c'])

        #ax.axvline(x=init, color='k', linestyle=':', linewidth=4)
        ax.tick_params(axis='x', labelsize=18)
        ax.tick_params(axis='y', labelsize=18)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    acqs = ['NEI', 'GIBBON', 'JES-e-LB2', 'WAAL']
    plot_hyperparameters(
        'results/20230106_gp_tasks',
        ['gp_2dim', 'gp_4dim'],
        acqs
    )
    #acqs = ['BQBC', 'QBMGP', 'BALM', 'WAAL-f', 'GIBBON']
    # plot_hyperparameters(
    #    'results/20221231_final_al',
    #   ['active_branin', 'active_hartmann6', 'ishigami', 'gramacy2'], acqs
    # )
