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
from pandas.errors import EmptyDataError

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

plt.rcParams['font.family'] = 'serif'

# Some constants for better estetics
# '#377eb8', '#ff7f00', '#4daf4a'
# '#f781bf', '#a65628', '#984ea3'
# '#999999', '#e41a1c', '#dede00'

BENCHMARK_PACKS = {
    'griewank_bo': {'names': ('griewank2_b4', 'griewank2_b16', 'griewank2_b64', 'griewank4_b4', 'griewank4_b16', 'griewank4_b64')},
    'michtest': {'names': ('mich2', 'mich5', 'mich10')},
    'synthetic_bo': {'names': ('branin', 'hartmann3', 'hartmann4', 'hartmann6', 'rosenbrock2', 'rosenbrock4')},
    'synthetic_hd': {'names': ('levy10', 'stybtang10', 'mich10')},
    'synthetic_bo_extra': {'names': ('rosenbrock2', 'rosenbrock4')},
    'hpo': {'names': ('hpo_segment', 'hpo_vehicle')},
    'xgb': {'names': ('xgb_segment', 'xgb_phoneme', 'xgb_kc1')},
    'pd1_scorebo': {'names': ('pd1_wmt', 'pd1_cifar', 'pd1_lm1b')},
    'init_bo': {'names': ('branin', 'hartmann3', 'hartmann4', 'hartmann6', 'rosenbrock2', 'stybtang4')},
    'noise_bo': {'names': ('branin', 'hartmann3', 'hartmann4', 'hartmann6', 'stybtang4', 'levy5'),
                 'noise': [5, 0.5, 0.5, 0.5, 5, 5, 10, 10]},
    'noise_bo_fixed': {'names': ('branin', 'hartmann3', 'hartmann4', 'hartmann6', 'stybtang4', 'levy5'),
                       'noise': [5, 1, 1, 1, 5, 5]},
    'synthetic_al': {'names': ('higdon', 'gramacy1', 'gramacy2', 'active_branin', 'ishigami', 'active_hartmann6')},
    'synthetic_al_rmse': {'names': ('higdon', 'gramacy1', 'gramacy2', 'active_branin', 'ishigami', 'active_hartmann6')},
    'gp_prior': {'names': (['gp_8dim'])},
    'saasbo': {'names': ('ackley4_25', 'hartmann6_25', 'lasso_dna'), 'best': (0, -3.3223, None)},
    'addgp_bo': {'names': ('gp_2_2_2dim', 'gp_2_2_2_2_2dim', 'cosmo')},
    'gtbo_syn': {'names': ('branin2', 'hartmann6', 'levy4')},
    'gtbo_real': {'names': ('lasso-dna', 'mopta08')},
    'colabo_syn': {'names': ('branin', 'hartmann3', 'hartmann6', 'ackley8', 'ackley4', 'rosenbrock4')},
    'colabo_paper': {'names': ('branin', 'hartmann4', 'ackley5', 'hartmann6', 'rosenbrock6')},
    'colabo_hard': {'names': ('hartmann4', 'levy5', 'hartmann6', 'rosenbrock6', 'stybtang7')},
    'colabo_debug': {'names': ('ackley3', 'ackley3')},
    'colabo_gp': {'names': ('gp_3dim', 'gp_5dim')},
    'ackleyabl': {'names': ('ackley5', 'ackley5')},
    'lcbench': {'names': (
        'lcbench189909',
        'lcbench167190',
        'lcbench168868',
        'lcbench126025',
        'lcbench167185',
        'lcbench189862',
        'lcbench189905',
        'lcbench189908',
        'lcbench168331',
        'lcbench126029',
        'lcbench189865',
        'lcbench167104',
        'lcbench167152',
        'lcbench167184',
        'lcbench189906',
        'lcbench167201',
        'lcbench189873',
        'lcbench168908',
        'lcbench167161',
        'lcbench167168',
        'lcbench168335',
        'lcbench167181',
        'lcbench167200',
        'lcbench167149',
        'lcbench7593',
        'lcbench146212',
        'lcbench168330',
        'lcbench34539',
        'lcbench168910',
        'lcbench189354',
        'lcbench3945',
        'lcbench189866',
        'lcbench126026',
        'lcbench168329'
    )},
    'pd1_colabo': {'names': ('pd1_wmt', 'pd1_cifar', 'pd1_lm1b')}
}


COLORS = {
    'MCpi-EI': 'black',
    'NEI_correct': 'k',
    'NEI_correct_name': 'k',
    'NEI_temp': '#984ea3',
    'NEI_wide': '#984ea3',
    'NEI_botorch': '#ff7f00',
    'NEI_MAP_AL_MAP': '#984ea3',
    'ScoreBO_J_HR_wide': 'navy',
    'ScoreBO_J_HR_botorch' : '#e41a1c',

    'ScoreBO_J': 'deeppink',
    'ScoreBO_J_HR_notrunc': 'red',
    'Scorebo_notrunc_MC': 'orange',
    'ScoreBO_M': 'limegreen',
    'JES-e-LB2': '#377eb8',
    'JES-e-LB2_AL_MAP': 'navy',
    'JES': 'goldenrod',
    'JES_2': 'dodgerblue',
    'JES-LB2': '#377eb8',
    'JES-e': 'navy',
    'JES-FB': 'crimson',
    'nJES-e': 'limegreen',
    'JESy': 'goldenrod',
    'JESy-e': 'darkgoldenrod',
    'MES': 'darkred',
    'EI-pi': '#e41a1c',
    'NEI-pi': 'deeppink',
    'EI': '#f781bf',
    'KG': 'crimson',
    'VES': 'dodgerblue',
    'NEI': 'k',
    'NEI_AL_MAP': 'orange',
    'Sampling': 'orange',
    # BETA TESTS

    'MCpi-EI_notrans': 'navy',
    'MCpi-KG': 'forestgreen',

    'UCB': 'orangered',
    'MCpi-UCB': 'lightsalmon',
    'WAAL': '#e41a1c',
    'GIBBON': '#4daf4a',
    'WAAL-f': '#e41a1c',
    'BALD': '#984ea3',
    'BALM': '#ff7f00',
    'QBMGP': '#377eb8',
    'BQBC': '#4daf4a',

    'ScoreBO_J_HR': '#e41a1c',
    'SCoreBO_J_HR48models': 'navy',
    'BOTorch_mean_prior': '#4daf4a',
    'Bad_prior': '#984ea3',
    'ALBO_prior': '#377eb8',
    'correct': '#ff7f00',
    'ScoreBO_warm': 'orange',
    'ScoreBO_large': 'dodgerblue',
    'ScoreBO_J_HR_BOinit': 'dodgerblue',
    'ScoreBO_J_HR_ALinit': 'limegreen',
    'ScoreBO_J_HR_MC_v4': 'dodgerblue',
    'ScoreBO_J_HR_MC_v4_512': 'navy',
    'ScoreBO_J_HR_MMfix': '#e41a1c',
    'SAL_HR': 'navy',
    'SAL_HR_default': 'navy',
    'SAL_JS': '#a65628',
    'SAL_WS': '#e41a1c',
    'NEI_MAP': 'orange',
    'TS_ana': 'purple',
    'NEI_no': 'dodgerblue',
    'SAL_WS_MC': 'dodgerblue',
    'ScoreBO_J_HR_notrunc_MC': 'dodgerblue',
    'ScoreBO_J_HR_notrunc_WS': 'navy',
    'NEI': 'k',
    'NEI_AL': 'darkgoldenrod',
    'JES-e-LB2_AL': 'navy',
    'JES_MAP': 'dodgerblue',
    'ScoreBO_J_HR_AL': 'purple',
    'PES': 'orange',
    'cma_es': '#377eb8',
    'random_search': '#ff7f00',
    'hesbo20': '',
    'hesbo10': '#f781bf',
    'alebo_101_': '#a65628',
    'alebo_201_': '#984ea3',
    'turbo-1-b1': '#999999',
    'turbo-5-b1': '#e41a1c',
    'saasbo': '#dede00',
    'gtbo': 'k',
    'baxus': '#4daf4a',

    'gtbo_turbo:False_reuse:False_gt:300': '#ff7f00',
    'gtbo_turbo:False_reuse:True_gt:300': 'darkgoldenrod',

    'gtbo_turbo:True_reuse:False_gt:200': '#ff7f00',
    'gtbo_turbo:True_reuse:False_gt:300': 'darkgoldenrod',
    'gtbo_turbo:True_reuse:True:200': 'forestgreen',
    'SCoreBO_J_HR_BQBC': '#a65628',
    'SCoreBO_J_HR_Squared': 'purple',
    'ScoreBO_J_JS': '#a65628',
    'ScoreBO_J_IG': 'deeppink',
    'JES-256': 'orange',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima2': 'orange',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima4': 'purple',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima8': 'red',  # Original
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima16': 'blue',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima32': 'blue',
    'ScoreBO_J_HR_model.num_samples64_algorithm.acq_kwargs.num_optima8': 'orange',
    'ScoreBO_J_HR_model.num_samples128_algorithm.acq_kwargs.num_optima8': 'purple',
    'ScoreBO_J_HR_model.num_samples512_algorithm.acq_kwargs.num_optima8': 'blue',
    'ScoreBO_J_HR_model.num_samples1024_algorithm.acq_kwargs.num_optima8': 'blue',
    'JES-e-LB2_': '#377eb8',
    'NEI_': 'k',
    'MCpi-LogEI': 'blue',
    'MCpi-MES': 'red',
    'MCpi-ucb_q1': 'blue',
    'MCpi-ucb_q3': 'red',
    'ScoreBO_J_HR_': '#e41a1c',
    'GIBBON_': '#4daf4a',
    'MCpi-LogEI_q1': 'yellow',
    'MCpi-LogEI_q2': 'darkgoldenrod',
    'MCpi-LogEI_q4': 'brown',
    'MCpi-LogEInoprior': 'goldenrod',
    'LogNEI': 'darkblue',
    'MCpi-EI_q1': 'limegreen',
    'MCpi-EI_q2': 'green',
    'MCpi-EI_q4': 'forestgreen',
    'MCpi-UCB_q1': 'orange',
    'MCpi-UCB_q2': 'red',
    'MCpi-UCB_q4': 'crimson',
    'MCpi-MES_q1': 'dodgerblue',
    'MCpi-MES_q2': 'blue',
    'MCpi-MES_q4': 'navy',
    'PiBO': '#4daf4a'

}


init = {
    'shekel4': 5,
    'levy6': 7,
    'xgb_segment': 5,
    'xgb_phoneme': 5,
    'xgb_kc1': 5,
    'griewank2_b4': 5,
    'griewank2_b16': 5,
    'griewank2_b64': 5,
    'griewank4_b4': 5,
    'griewank4_b16': 5,
    'griewank4_b64': 5,
    'ackley3': 4,
    'ackley3_n05': 4,
    'stybtang7': 8,
    'cosine8': 9,
    'active_hartmann6': 100,
    'stybtang10': 11,
    'gp_2_2_2_2dim': 3,
    'gp_2_2_2dim': 5,
    'mich10': 11,
    'mich2': 3,
    'gp_2_2dim': 3,
    'botorch_3_3_2dim': 0,
    'stybtang4': 5,
    'stybtang6': 7,
    'branin': 3,
    'branin_25': 3,
    'hartmann3': 4,
    'hartmann6': 7,
    'hartmann6_25': 5,
    'ackley8': 9,
    'levy5': 6,
    'levy8': 9,
    'ackley4': 5,
    'lasso_dna': 5,
    'ackley5': 6,
    'ackley4_25': 5,
    'alpine5': 6,
    'rosenbrock4': 5,
    'rosenbrock4_25': 5,
    'rosenbrock8': 9,
    'rosenbrock12': 13,
    'mich5': 6,
    'levy12': 13,
    'xgboost': 9,
    'fcnet': 7,
    'gp_2dim': 3,
    'gp_4dim': 5,
    'gp_6dim': 7,
    'gp_8dim': 9,
    'gp_12dim': 13,
    'active_branin': 3,
    'gramacy1': 2,
    'gramacy2': 3,
    'higdon': 2,
    'ishigami': 4,
    'hartmann4': 5,
    'rosenbrock2': 3,
    'hpo_blood': 5,
    'hpo_segment': 5,
    'hpo_australian': 5,
    'botorch_3_3_4dim': 5,
    'ackley12': 13,
    'gp_2_2_2_2_2dim': 5,
    'gp_1_1_4_4dim': 5,
    'gp_1_1_2_4_8dim': 5,
    'gramacy2': 3,
    'gramacy1': 3,
    'higdon': 3,
    'ishigami': 3,
    'active_hartmann6': 3,
    'active_branin': 3,
    'cosmo': 3,
    'rosenbrock20': 20 + 1,
    'rastrigin10': 10 + 1,
    'rastrigin20': 20 + 1,
    'levy10': 10 + 1,
    'levy16': 16 + 1,
    'hartmann12': 12 + 1,
    'ackley10': 10 + 1,
    'ackley16': 16 + 1,
    'hartmann6': 10,
    'levy4': 10,
    'branin2': 10,
    'lasso-dna': 10,
    'mopta08': 10,
    'svm': 10,
    'rosenbrock6': 7,
    'hpo_vehicle': 5,
    'hpo_segment': 5,
    'hpo_blood': 5,
    'pd1_lm1b': 12,
    'pd1_wmt': 12,
    'pd1_cifar': 12,
    'pd1_image': 12,
    'pd1_uniref': 12,

    'lcbench189909': 5,
    'lcbench167190': 5,
    'lcbench168868': 5,
    'lcbench126025': 5,
    'lcbench167185': 5,
    'lcbench189862': 5,
    'lcbench189905': 5,
    'lcbench189908': 5,
    'lcbench168331': 5,
    'lcbench126029': 5,
    'lcbench189865': 5,
    'lcbench167104': 5,
    'lcbench167152': 5,
    'lcbench167184': 5,
    'lcbench189906': 5,
    'lcbench167201': 5,
    'lcbench189873': 5,
    'lcbench168908': 5,
    'lcbench167161': 5,
    'lcbench167168': 5,
    'lcbench168335': 5,
    'lcbench167181': 5,
    'lcbench167200': 5,
    'lcbench167149': 5,
    'lcbench7593': 5,
    'lcbench146212': 5,
    'lcbench168330': 5,
    'lcbench34539': 5,
    'lcbench168910': 5,
    'lcbench189354': 5,
    'lcbench3945': 5,
    'lcbench189866': 5,
    'lcbench126026': 5,
    'lcbench168329': 5

}


NAMES = {
    'SAL_HR': 'SAL - HR',
    'SAL_HR_32': 'SAL - HR',
    'SAL_JS': 'SAL - JS',
    'TS_ana': 'TS - analytic',
    'SAL_WS': 'SAL - WS',
    'JES_ben': 'JES-$y$',
    'ScoreBO_J': 'ScoreBO_J',
    'ScoreBO_M': 'ScoreBO_M',
    'GIBBON': 'GIBBON',
    'JES': 'JES',
    'JES-e': '$f$-JES-\u03B3',
    'JES-FB': 'FB-JES',
    'JES-e-LB2': 'JES',
    'PES': 'PES',
    'JES-e-LB2AL': 'JES - AL',
    'JES-LB2': 'LB2-JES',
    'nJES': 'newJES',
    'nJES-e': 'newJES-\u03B3',
    'JESy': '$y$-JES',
    'JESy-e': '$y$-JES-\u03B3',
    'JES-e-pi': '\u03B3-JES-pi',
    'MES': 'MES',
    'EI': 'EI',
    'NEI': 'EI',
    'JES-pi': 'JES-pi',
    'EI-pi': 'EI-pi',
    'NEI-pi': 'Noisy EI-pi',
    'JES-e01': '\u03B3-0.1-JES',
    'JES-e01-pi': '\u03B3-0.1-JES-pi',
    'Sampling': 'Prior Sampling',
    # BETA TESTS
    'JES-e-pi-2': '\u03B3-JES-pi-2',
    'JES-e-pi-5': '\u03B3-JES-pi-5',
    'JES-e-pi-20': '\u03B3-JES-pi-20',
    'JES-e-pi-50': '\u03B3-JES-pi-50',
    'VES': 'VES',
    'KG': 'KG',
    'NEI_MAP': 'EI - MAP',
    'NEI_AL': 'Noisy EI - AL init',
    'NEI_no': 'EI - sobol init',
    'NEI_temp': 'EI',
    'NEI_correct': 'EI - Correct HPs',
    'NEI_correct_name': 'EI',

    'MCpi-EI': 'MCpi-EI',
    'NEI_wide': 'EI - Wide LogNormal Prior',
    'NEI_botorch': 'EI - BoTorch Prior',
    'JES-256': 'JES-256',

    'WAAL': 'SAL-WS',
    'WAAL-f': 'SAL-WS',
    'BALD': 'BALD',
    'BALM': 'BALM',
    'QBMGP': 'QBMGP',
    'BQBC': 'BQBC',
    'MCpi-LogEInoprior': 'MCpi-LogEInoprior',
    'MCpi-LogEInoprior_acqnorm': 'MCpi-LogEInoprior_acqnorm',
    'ScoreBO_M_HR': 'ScoreBO_M - HR',
    'ScoreBO_M_BC': 'ScoreBO_M - BC',
    'ScoreBO_M_JS': 'ScoreBO_M - JS',
    'ScoreBO_M_WS': 'ScoreBO_M - WS',
    'ScoreBO_M_HR_star': 'ScoreBO*_M - HR',
    'ScoreBO_M_BC_star': 'ScoreBO*_M - BC',
    'ScoreBO_M_JS_star': 'ScoreBO*_M - JS',
    'ScoreBO_M_WS_star': 'ScoreBO*_M - WS',
    'ScoreBO_J_HR': 'SCoreBO',
    'ScoreBO_J_HR_notrunc': 'SCoreBO',
    'ScoreBO_J_HR_notrunc_MC': 'SCoreBO - HR',
    'ScoreBO_J_HR_notrunc_WS': 'ScoreBO - WS',
    'Scorebo_notrunc_MC': 'SCoreBO - MC',

    'ScoreBO_J_HR_wide': 'SCoreBO - Wide LogNormal',
    'ScoreBO_J_HR_botorch' : 'SCoreBO - BoTorch',
    'SAL_HR_default': 'SAL - HR',
    'JES_2': 'JES_2',
    'ScoreBO_J_noinit_HR': 'ScoreBO_J - HR',
    'Bad_prior': 'Bad prior',
    'BOTorch_mean_prior': 'BOTorch prior',
    'ALBO_prior': 'Good prior',
    'correct': 'EI - Correct HPs',
    'SAL_WS_MC': 'SAL - MC',
    'JES-e-LB2ben': 'JES - $y*$',

    'JES-e-LB2_AL_MAP': 'JES - MAP',
    'NEI_AL_MAP': 'NEI - MAP',
    'SCoreBO_J_HR48models': 'SCoreBO - 48 models',
    'SCoreBO_J_HR_BQBC': 'SCoreBO - Mean',
    'SCoreBO_J_HR_Squared': 'SCoreBO - Squared',
    'ScoreBO_J_JS': 'ScoreBO - JS',
    'ScoreBO_J_IG': 'ScoreBO - Info Gain',

    # GTBO stuff:
    'cma_es': 'CMA-ES',
    'random_search': 'Random Search',
    'baxus': 'BaXUS',
    'hesbo10': 'HeSBO-10',
    'alebo_101_': 'ALEBO-10',
    'alebo_201_': 'ALEBO-20',
    'turbo-1-b1': 'TuRBO-1',
    'turbo-5-b1': 'TuRBO-5',
    'saasbo': 'SAASBO',
    'gtbo': 'GTBO',
    'LogNEI': 'LogEI',
    'gtbo_turbo:False_reuse:False_gt:300': 'GTBO',
    'gtbo_turbo:False_reuse:True_gt:300': 'GTBO-Re',


    'gtbo_turbo:True_reuse:False_gt:200': 'GTBO-200',
    'gtbo_turbo:True_reuse:False_gt:300': 'GTBO-300',
    'gtbo_turbo:True_reuse:True:200': 'GTBO-Re-200',
    'jes_rff16k': 'RFF-16',
    'jes_lowopt': 'Low opt',
    'ScoreBO_J_HR_': 'SCoreBO',
    'JES-e-LB2_': 'JES',
    'GIBBON_': 'GIBBON',
    'NEI_': 'Noisy EI',
    'NEI_MAP_AL_MAP': 'EI - MAP',
    # Ablations
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima2': '256 Samples, 2 optima',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima4': '256 Samples, 4 optima',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima8': '256 Samples, 8 optima',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima16': '256 Samples, 16 optima',
    'ScoreBO_J_HR_model.num_samples256_algorithm.acq_kwargs.num_optima32': '256 Samples, 32 optima',
    'ScoreBO_J_HR_model.num_samples64_algorithm.acq_kwargs.num_optima8': '64 Samples, 8 optima',
    'ScoreBO_J_HR_model.num_samples128_algorithm.acq_kwargs.num_optima8': '128 Samples, 8 optima',
    'ScoreBO_J_HR_model.num_samples512_algorithm.acq_kwargs.num_optima8': '512 Samples, 8 optima',
    'ScoreBO_J_HR_model.num_samples1024_algorithm.acq_kwargs.num_optima8': '1024 Samples, 8 optima',
    'MCpi-LogEI_q1': 'MCpi-LogEI_q1',
    'MCpi-LogEI_q2': 'MCpi-LogEI_q2',
    'MCpi-LogEI_q4': 'MCpi-LogEI_q4',
    'MCpi-EI_q1': 'MCpi-EI_q1',
    'MCpi-EI_q2': 'MCpi-EI_q2',
    'MCpi-EI_q4': 'MCpi-EI_q4',
    'MCpi-UCB_q1': 'MCpi-UCB_q1',
    'MCpi-UCB_q2': 'MCpi-UCB_q2',
    'MCpi-UCB_q4': 'MCpi-UCB_q4',
    'MCpi-MES_q1': 'MCpi-MES_q1',
    'MCpi-MES_q2': 'MCpi-MES_q2',
    'MCpi-MES_q4': 'MCpi-MES_q4',
    'MCpi-LogEI': 'ColaBO-LogEI',
    'MCpi-MES': 'ColaBO-MES',
    'MCpi-LogEInoprior_acqnorm_q1': 'MCpi-LogEInoprior_acqnorm_q1',
    'MCpi-LogEInoprior_acqnorm_q3': 'MCpi-LogEInoprior_acqnorm_q3',
    'MCpi-ucb_acqnorm_q1': 'MCpi-ucb_q1',
    'MCpi-ucb_acqnorm_q3': 'MCpi-ucb_q3',
    'PiBO': 'PiBO'

}

PLOT_LAYOUT = dict(
    linewidth=1.5,
    markevery=10,
    markersize=6,
    markeredgewidth=4
)

MARKERS = {
    'MCpi-LogEI': '*',
    'MCpi-MES': '*',
    'PiBO': '*',
}

BENCHMARK_NAMES = {

    'levy6': 'Levy (6D)',
    'levy5': 'Levy (5D)',
    'ackley3': 'Ackley-3',
    'ackley3_n05': 'Ackley-3, noise',
    'levy8': 'Levy (8D)',

    'hartmann3': 'Hartmann (3D)',  # ,   $\sigma_\\varepsilon = 0.5$',
    'stybtang4': 'Styblinski-Tang (4D),   $\sigma_\\varepsilon = 5$',
    'stybtang6': 'Styblinski-Tang-6',
    'hartmann4': 'Hartmann (4D)',  # ,   $\sigma_\\varepsilon = 0.5$',
    'hartmann6': 'Hartmann (6D)',  # ,   $\sigma_\\varepsilon = 0.5$',
    'hartmann6_25': 'Hartmann-6 (25D)',
    'rosenbrock2': 'Rosenbrock (2D)',
    'rosenbrock4': 'Rosenbrock (4D)',
    'rosenbrock8': 'Rosenbrock-8',
    'rosenbrock12': 'Rosenbrock-12',
    'ackley8': 'Ackley (8D)',
    'ackley12': 'Ackley-12',
    'ackley16': 'Ackley-16',
    'ackley4': 'Ackley (4D)',
    'ackley5': 'Ackley-5',
    'levy12': 'Levy-12',
    'levy16': 'Levy-16',
    'rosenbrock4_25': 'Rosenbrock-4 / 25D',
    'rosenbrock6': 'Rosenbrock (6D)',
    'branin': 'Branin (2D)',  # ,   $\sigma_\\varepsilon = 5$',

    'stybtang7': 'Stybtang (7D)',
    'cosine8': 'Cosine (8D)',
    'griewank2_b4': 'Griewank - (2D) B4',
    'griewank2_b16': 'Griewank - (2D) B16',
    'griewank2_b64': 'Griewank - (2D) B64',
    'griewank4_b4': 'Griewank - (4D) B4',
    'griewank4_b16': 'Griewank - (4D) B16',
    'griewank4_b64': 'Griewank - (4D) B64',

    'branin_25': 'Branin / 25D',
    'ackley4_25': 'Ackley-4 (25D)',
    'gp_3dim': 'GP-sample (3D)',
    'gp_4dim': 'GP-sample (4D)',
    'gp_5dim': 'GP-sample (5D)',
    'gp_8dim': 'GP-sample (8D)',
    'gp_12dim': 'GP-sample (12D)',
    'hpo_blood': 'hpo_blood',
    'hpo_segment': 'hpo_segment',
    'hpo_australian': 'hpo_australian',
    'hpo_vehicle': 'hpo_vehicle',
    'gp_2_2dim': 'GP-sample (2D+2D)',
    'botorch_3_3_2dim': 'GP-sample (3D+3D+2D)',
    'botorch_3_3_4dim': 'GP-sample (3D+3D+4D)',
    'gp_2_2_2_2dim': 'GP-sample (2D+2D+2D+2D)',
    'gp_2_2_2_2_2dim': 'GP-sample (2D+2D+2D+2D+2D)',
    'gp_2_2_2dim': 'GP-sample (2D+2D+2D)',
    'gp_1_1_4_4dim': 'GP-sample (1D+1D+4D+4D)',
    'gp_1_2_3_4dim': 'GP-sample (1D+2D+3D+4D)',
    'lasso_dna': 'Lasso-DNA (180D)',
    'gramacy2': 'Gramacy (2D)',
    'gramacy1': 'Gramacy (1D)',
    'higdon': 'Higdon (1D)',
    'mich2': 'Mich (2D)',
    'mich5': 'Mich (5D)',
    'mich10': 'Mich (10D)',
    'ishigami': 'Ishigami (3D)',
    'active_hartmann6': 'Hartmann (6D)',
    'active_branin': 'Branin (2D)',
    'gp_1_1_2_4_8dim': 'GP(1D+1D+2D+2D+4D+8D',
    'cosmo': 'Cosmological Constants (11D)',
    'rosenbrock20': 'Rosenbrock-20',
    'rastrigin10': 'Rastrigin-10',
    'rastrigin20': 'Rastrigin-20',
    'levy10': 'Levy-10',
    'hartmann12': 'Hartmann-12',
    'ackley10': 'Ackley-10',
    'stybtang10': 'Styblinski-Tang-10',
    'levy4': 'Levy (4D) - 300D Embedding',
    'branin2': 'Branin (2D) - 300D Embedding',
    'lasso-dna': 'Lasso-DNA',
    'mopta08': 'MOPTA08',
    'svm': 'SVM',
    'xgb_segment': 'segment',
    'xgb_phoneme': 'phoneme',
    'xgb_kc1': 'kc1',
    'pd1_lm1b': 'pd1-lm1b'.upper(),
    'pd1_wmt': 'pd1-wmt'.upper(),
    'pd1_cifar': 'pd1-cifar'.upper(),
    'pd1_image': 'pd1-image'.upper(),
    'pd1_uniref': 'pd1-uniref'.upper(),
    'shekel4': 'Shekel (4D)',
    'lcbench189909': 'LCBench-189909',
    'lcbench167190': 'LCBench-167190',
    'lcbench168868': 'LCBench-168868',
    'lcbench126025': 'LCBench-126025',
    'lcbench167185': 'LCBench-167185',
    'lcbench189862': 'LCBench-189862',
    'lcbench189905': 'LCBench-189905',
    'lcbench189908': 'LCBench-189908',
    'lcbench168331': 'LCBench-168331',
    'lcbench126029': 'LCBench-126029',
    'lcbench189865': 'LCBench-189865',
    'lcbench167104': 'LCBench-167104',
    'lcbench167152': 'LCBench-167152',
    'lcbench167184': 'LCBench-167184',
    'lcbench189906': 'LCBench-189906',
    'lcbench167201': 'LCBench-167201',
    'lcbench189873': 'LCBench-189873',
    'lcbench168908': 'LCBench-168908',
    'lcbench167161': 'LCBench-167161',
    'lcbench167168': 'LCBench-167168',
    'lcbench168335': 'LCBench-168335',
    'lcbench167181': 'LCBench-167181',
    'lcbench167200': 'LCBench-167200',
    'lcbench167149': 'LCBench-167149',
    'lcbench7593': 'LCBench-7593',
    'lcbench146212': 'LCBench-146212',
    'lcbench168330': 'LCBench-168330',
    'lcbench34539': 'LCBench-34539',
    'lcbench168910': 'LCBench-168910',
    'lcbench189354': 'LCBench-189354',
    'lcbench3945': 'LCBench-3945',
    'lcbench189866': 'LCBench-189866',
    'lcbench126026': 'LCBench-126026',
    'lcbench168329': 'LCBench-168329'
}


def get_gp_regret(gp_task_type):
    gp_task_files = glob(join(dirname(abspath(__file__)),
                              f'gp_sample/{gp_task_type}/*.json'))
    opts = {}
    for file in sorted(gp_task_files):
        name = file.split('/')[-1].split('_')[-1].split('.')[0]
        with open(file, 'r') as f:
            opt_dict = json.load(f)
            opts[name] = opt_dict['opt']
    return opts


def get_regret(benchmark):
    if 'gp_' in benchmark:
        gp_regrets = get_gp_regret(benchmark)
        for key in gp_regrets.keys():
            gp_regrets[key] = gp_regrets[key] + 0.18
        return gp_regrets

    else:
        regrets = {
            'shekel4': 10.17,
            'gp_12dim': 0,
            'branin2': -0.397887,

            'branin': -0.397887,
            'branin_25': -0.397887,
            'hartmann3': 3.86278,
            'hartmann6': 3.32237,
            'hartmann6_25': 3.32237,
            'hartmann4': 3.1344945430755615,
            'ackley4': 0,
            'ackley3': 0,
            'ackley3_n05': 0,
            'ackley5': 0,
            'ackley4_25': 0,
            'mich10': 9,

            'stybtang7': 39.16599 * 7,
            'cosine8': -0.8,
            'ackley8': 0,
            'ackley12': 0,
            'alpine5': 0,
            'rosenbrock12': 0,
            'rosenbrock4': 0,
            'rosenbrock6': 0,
            'rosenbrock4_25': 0,
            'rosenbrock2': 0,
            'rosenbrock8': 0,
            'gp_2dim': 0,
            'gp_4dim': 0,
            'gp_6dim': 0,
            'levy12': 0,
            'fcnet': 0.03 + np.exp(-5) + np.exp(-6.6) ,
            'xgboost': -(8.98 + 2 * np.exp(-6)),

            'active_branin': 0,
            'gramacy1': 0,
            'gramacy2': -0.428882,
            'higdon': 0,
            'ishigami': 10.740093895930428 + 1e-3,
            'active_hartmann6': 3.32237,
            'gp_2_2dim': 0,
            'botorch_3_3_2dim': 0,
            'botorch_2_2_2_2dim': 0,
            'botorch_3_3_4dim': 0,
            'gp_2_2_2_2dim': 0,
            'gp_2_2_2dim': 0,
            'lasso_dna': None,
            'mich5': 4.687658,
            'gp_2_2_2_2_2dim': 0,
            'gp_1_1_2_4_8dim': 5,
            'rosenbrock20': 0,
            'rastrigin10': 0,
            'rastrigin20': 0,
            'stybtang10': 39.16599 * 10,
            'stybtang4': 39.16599 * 4,
            'stybtang6': 39.16599 * 6,
            'levy10': 0,
            'levy5': 0,
            'levy8': 0,
            'levy16': 0,
            'hartmann12': 2 * 3.32237,
            'ackley10': 0,
            'ackley16': 0,

            'griewank2_b4': 0,
            'griewank2_b16': 0,
            'griewank2_b64': 0,
            'griewank4_b4': 0,
            'griewank4_b16': 0,
            'griewank4_b64': 0,

        }
        return regrets.get(benchmark, False)


def process_funcs_args_kwargs(input_tuple):
    '''
    helper function for preprocessing to assure that the format of (func, args, kwargs is correct)
    '''
    if len(input_tuple) != 3:
        raise ValueError(
            f'Expected 3 elements (callable, list, dict), got {len(input_tuple)}')

    if not callable(input_tuple[0]):
        raise ValueError('Preprocessing function is not callable.')

    if type(input_tuple[1]) is not list:
        raise ValueError('Second argument to preprocessing function is not a list.')

    if type(input_tuple[2]) is not dict:
        raise ValueError('Third argument to preprocessing function is not a dict.')

    return input_tuple


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
            run_paths = glob(join(acq_path, '*[0-9].csv'))
            paths_dict[benchmark_name][acq_name] = sorted(run_paths)

    return paths_dict


def get_dataframe(paths, funcs_args_kwargs=None, idx=0):
    '''
    For a given benchmark and acquisition function (i.e. the relevant list of paths),
    creates the dataframe that includes the relevant metrics.

    Parameters:
        paths: The paths to the experiments that should be included in the dataframe
        funcs_args_kwargs: List of tuples of preprocessing arguments,
    '''
    # ensure we grab the name from the right spot in the file structure
    names = [path.split('/')[-1].split('.')[0] for path in paths]

    # just create the dataframe and set the column names
    complete_df = pd.DataFrame(columns=names)

    # tracks the maximum possible length of the dataframe
    max_length = None

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for path, name in zip(paths, names):
            per_run_df = pd.read_csv(path)
            # this is where we get either the predictions or the true values
            if funcs_args_kwargs is not None:
                for func_arg_kwarg in funcs_args_kwargs:
                    func, args, kwargs = process_funcs_args_kwargs(func_arg_kwarg)
                    # try:
                    per_run_df = func(per_run_df, name, *args, **kwargs)
            complete_df.loc[:, name] = per_run_df.iloc[:, 0]
    return complete_df


def get_min(df, run_name, metric, minimize=True):
    if 'gp_' in run_name or 'lasso' in run_name:
        minimize = True
    if metric not in ['True Eval', 'Guess values']:
        df[metric] = -df[metric]

    min_observed = np.inf
    mins = np.zeros(len(df))
    for r, row in enumerate(df[metric]):
        if minimize:
            if (row < min_observed) and row != 0:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed and row != 0:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def get_min_flip100(df, run_name, metric, minimize=True):

    df[metric] = 100 - df[metric]

    min_observed = np.inf
    mins = np.zeros(len(df))
    for r, row in enumerate(df[metric]):
        if minimize:
            if (row < min_observed) and row != 0:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed and row != 0:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def get_min_flip1(df, run_name, metric, minimize=True):

    df[metric] = 1 - df[metric]

    min_observed = np.inf
    mins = np.zeros(len(df))
    for r, row in enumerate(df[metric]):
        if minimize:
            if (row < min_observed) and row != 0:
                min_observed = row
            mins[r] = min_observed
        else:
            if -row < min_observed and row != 0:
                min_observed = -row
            mins[r] = min_observed
    return pd.DataFrame(mins, columns=[run_name])


def get_metric(df, run_name, metric, minimize=True):
    nonzero_elems = df[metric][df[metric] != 0].to_numpy()
    first_nonzero = nonzero_elems[0]
    num_to_append = np.sum(df[metric] == 0)
    result = np.append(np.ones(num_to_append) * first_nonzero, nonzero_elems)

    if metric == 'RMSE':
        result = np.log10(result)
    return pd.DataFrame(result, columns=[run_name])


def compute_regret(df, run_name, regret, log=True):
    if type(regret) is dict:
        run_name_short = ''.join(run_name.split('_')[-2:])
        regret = regret[run_name_short]

    if np.any(df.iloc[:, 0] + regret) < 0:
        vals = df.iloc[:, 0]
        error_msg = f'Regret value: {regret}, best observed {-vals[vals + regret < 0]} for run {run_name_short }.'
        f'Re-optimize GP.'
        raise ValueError(error_msg)

    if log:
        mins = df.iloc[:, 0].apply(lambda x: np.log10(x + regret))
    else:
        mins = df.iloc[:, 0].apply(lambda x: x + regret)
    return pd.DataFrame(mins)


def compute_nothing(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: x)
    return pd.DataFrame(mins)


def compute_flip100(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: 100 - x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: 100 - x)
    return pd.DataFrame(mins)


def compute_negative(df, run_name, regret, log=True):
    if log:
        mins = df.iloc[:, 0].apply(lambda x: x)
    else:
        mins = df.iloc[:, 0].apply(lambda x: x)

    return pd.DataFrame(-mins)


def get_empirical_regret(data_dicts, metric, idx=0, log_coeff=0.13):
    MAX_REGRET = -1e10
    regrets = None
    for data_dict in [data_dicts]:

        for run_name, files in data_dict.items():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                names = [f'run' + file.split('/')[-1].split('_')
                         [-1].split('.')[0] for file in files]

                # just create the dataframe and set the column names
                if regrets is None:
                    regrets = {f'{name}': MAX_REGRET for name in names}
                for path, name in zip(files, names):
                    per_run_df = pd.read_csv(path).loc[:, metric]
                    # this is where we get either the predictions or the true values

                    if per_run_df.max() > regrets[name]:
                        regrets[name] = -per_run_df.max()

    return {name: -regret + log_coeff for name, regret in regrets.items()}


def compute_significance(reference, data_dict):
    from scipy.stats import ttest_ind
    for benchmark in data_dict.keys():

        reference_data = data_dict[benchmark].pop(reference, None)
        if reference is None:
            print('The provided reference', reference, 'does not exist in the data array.'
                  'Available runs are', list(data_dict.keys()), '. Cannot compute significance.')
            return
        ref_stats = reference_data[:, -1]
        for comp, vals in data_dict[benchmark].items():
            comp_stats = vals[:, -1]
            res = ttest_ind(ref_stats, comp_stats, alternative='less')
            print(benchmark, reference, comp, res)


def compute_ranking(all_data, ax, plot_config, legend=False):
    # num_steps x num_runs x num_methods
    maxlen = 0
    ranking_data = {}
    for benchmark in all_data.keys():
        print(benchmark)
        num_acqs = len(all_data[benchmark])
        benchmark_lengths = all_data[benchmark][list(
            all_data[benchmark].keys())[0]].shape[0]
        num_runs = plot_config['num_runs']
        benchmark_data = np.empty((benchmark_lengths, num_runs, num_acqs))

        for i, (acq, data) in enumerate(all_data[benchmark].items()):
            print(acq, benchmark)
            benchmark_data[..., i] = data[:, :num_runs]
        # ranking = benchmark_data.mean(1).argsort(axis=-1)
        ranking = benchmark_data.argsort(axis=-1)
        maxlen = max(maxlen, benchmark_lengths)
        ranking_data[benchmark] = ranking

    num_benchmarks = len(all_data)
    num_runs = plot_config['num_runs']
    extended_rankings = np.zeros((maxlen, num_benchmarks * num_runs, num_acqs))
    for bench_idx, bench_data in enumerate(ranking_data.values()):
        print(bench_idx, bench_data.shape)
        wrong_ratio = len(bench_data) / maxlen
        for i in range(maxlen):
            # print(bench_idx, i, i * wrong_ratio, ben   ch_data.shape)
            # extended_rankings[i, bench_idx, :] = bench_data[int(i * wrong_ratio), :]
            extended_rankings[i, bench_idx
                              * num_runs: (bench_idx + 1) * num_runs, :] = bench_data[int(i * wrong_ratio), :]
    acq_order = all_data[benchmark].keys()
    for idx, acq in enumerate(acq_order):
        # if acq == 'GIBBON':
        #    acq_sub = 'JES-e-LB2'
        # elif acq == 'JES-e-LB2':
        #    acq_sub = 'GIBBON'

        # elif acq == 'BQBC':
        #    acq_sub = 'BALM'
        # elif acq == 'BALM':
        #    acq_sub = 'BQBC'
        # else:
        acq_sub = acq
        acq_ranks = extended_rankings.mean(1)[..., idx]
        MA = 10
        acq_ranks = np.convolve(acq_ranks, np.ones(MA), 'valid') / MA
        ax.plot(np.linspace(MA, 100, maxlen - MA + 1),
                acq_ranks + 1, label=NAMES[acq_sub], color=COLORS[acq_sub], linewidth=2)

    # for idx, acq in enumerate(acq_order):
    #    plt.plot(np.linspace(0, 100, len(
    #        ranking_data['hartmann6'])), ranking_data['hartmann6'][:, 20:30][..., idx], label=NAMES[acq], color=COLORS[acq])
        # ax.legend(fontsize=plot_config['fontsizes']['legend'])
    #ax.set_xlabel('Percentage of run', fontsize=plot_config['fontsizes']['metric'])
    ax.tick_params(axis='x', labelsize=15)

    import matplotlib.ticker as mtick
    fmt = '%.0f%%'  # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    ax.xaxis.set_major_formatter(xticks)
    ax.set_xlabel('Percentage of run', fontsize=plot_config['fontsizes']['metric'])
    ax.tick_params(axis='y', labelsize=15)
    ax.set_title('Relative ranking',
                 fontsize=plot_config['fontsizes']['benchmark_title'])
    if legend:
        ax.legend(fontsize=14)


SIZE_CONFIGS = {
    # GTBO:
    'gtbo_syn':
    {
        'reference_run': 'SAASBO',
        'subplots':
        {
            'figsize': (20, 5.1),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 21,
            'metric': 18,
            'iteration': 15,
            'legend': 17,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Simple Regret',
        'metric': 'f',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,
    },
        'gtbo_real':
    {
        'reference_run': 'SAASBO',
        'subplots':
        {
            'figsize': (20, 5.1),
            'nrows': 1,
            'ncols': 2,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 21,
            'metric': 18,
            'iteration': 15,
            'legend': 17,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best observed value',
        'metric': 'f',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,
    },



    'synthetic_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 25,
        'subplots':
        {
            'figsize': (20, 8.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'synthetic_hd':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 25,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 9,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'True Eval',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,

    },

    'hpo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 10,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best Observed Value',
        'metric': 'metric_name',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': True,

    },
    'lcbench':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 10,
        'subplots':
        {
            'figsize': (35, 20.5),
            'nrows': 7,
            'ncols': 5,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best Observed Value',
        'metric': 'True Eval',
        'compute': compute_nothing,
        'get_whatever': get_min_flip100,
        'flip': True,
        'log_regret': True,

    },
    'xgb':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 10,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best Observed Value',
        'metric': 'metric_name',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': True,

    },

    'pd1':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 20,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Best Observed Value',
        'flip': True,
        'split_range':
        {
            'pd1_lm1b': ((62, 80), (60.2, 62)),
            'pd1_wmt': ((32, 80), (28.7, 32)),
            'pd1_cifar': ((19.8, 80), (17.6, 19.8)),

        },
        'percentage': True,
        'metric': 'metric_name',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': True,

    },

    'synthetic_bo_extra':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 7.5),
            'nrows': 2,
            'ncols': 4,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'griewank_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'local_MLL',
        'compute': compute_nothing,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'init_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },

    'colabo_syn':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': False,

    },
    'colabo_paper':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 5,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,

    },
    'colabo_hard':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 5,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,

    },
    'colabo_debug':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 2,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,

    },
    'pd1_colabo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 10,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_nothing,
        'get_whatever': get_min_flip1,
        'log_regret': True,

    },

    'colabo_gp':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 2,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,

    },
    'ackleyabl':
    {
        'reference_run': 'ScoreBO_J_HR',
        'num_runs': 50,
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 2,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': True,

    },

    'noise_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 8.5),
            'nrows': 2,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'noise_bo_fixed':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 2,
            'ncols': 5,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'Guess values',
        'compute': compute_regret,
        'get_whatever': get_metric,
        'log_regret': True,

    },
    'michtest':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log Inference Regret',
        'metric': 'True Eval',
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': True,

    },
    'synthetic_al':
    {
        'reference_run': 'SAL_HR',
        'num_runs': 25,
        'subplots':
        {
            'figsize': (20, 6.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
            'h_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 16,
            'legend': 15,
        },
        'plot_args':
        {
            'infer_ylim': True,
            'start_at': 20
        },
        'metric_name': 'Negative MLL',
        'metric': 'MLL',
        'compute': compute_nothing,
        'get_whatever': get_metric,
        'log_regret': False,

    },
    'synthetic_al_rmse':
    {
        'reference_run': 'SAL_HR',
        'subplots':
        {
            'figsize': (20, 5.5),
            'nrows': 1,
            'ncols': 6,
        },
        'tight_layout':
        {
            'w_pad': 0.08,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 16,
            'iteration': 0,
            'legend': 15,
        },
        'plot_args':
        {
        },
        'metric_name': 'Log RMSE',
        'metric': 'RMSE',
        'compute': compute_nothing,
        'get_whatever': get_metric,
        'log_regret': False,
            'legend': 15,

    },

    'gp_prior':
    {
        'reference_run': 'ScoreBO_J_HR',
        'empirical_regret': False,
        'subplots':
        {
            'figsize': (7, 4),
            'nrows': 1,
            'ncols': 1,
        },
        'tight_layout':
        {
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 15,
            'legend': 16,
        },
        'plot_args':
        {
            'start_at': 0,
        },
        'metric_name': 'Simple Regret',
        'metric': 'True Eval',
        'compute': compute_regret,
        'get_whatever': get_min,
        'log_regret': False,

    },
    'saasbo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (20, 3.6),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.75,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 0,
            'legend': 16,
        },
        'plot_args':
        {
            'start_at': 1,
            'init': 3,
        },
        'metric_name': 'Min. Observed Value',
        'metric': ('ackley4_25', 'hartmann6_25', 'lasso_dna'),
        'compute': compute_nothing,
        'get_whatever': get_min,
        'log_regret': False,


    },
    'addgp_bo':
    {
        'reference_run': 'ScoreBO_J_HR',
        'subplots':
        {
            'figsize': (15, 4.5),
            'nrows': 1,
            'ncols': 3,
        },
        'tight_layout':
        {
            'w_pad': 0.75,
        },
        'fontsizes':
        {
            'benchmark_title': 20,
            'metric': 18,
            'iteration': 0,
            'legend': 16,
        },
        'plot_args':
        {
        },
        'metric_name': ('Simple Regret', 'Simple Regret', 'Min. Observed Value'),
        'metric': 'True Eval',
        'compute': (compute_regret, compute_regret, compute_nothing),
        'get_whatever': get_min,
        'log_regret': False,
        'empirical_regret': True,
    },


}


def plot_optimization(data_dict,
                      preprocessing=None,
                      title='benchmark',
                      path_name='',
                      linestyle='solid',
                      xlabel='X',
                      ylabel='Y',
                      use_empirical_regret=False,
                      fix_range=None,
                      start_at=0,
                      only_plot=-1,
                      flip=False,
                      percentage=False,
                      show_xticks=True,
                      names=None,
                      predictions=False,
                      init=2,
                      n_markers=20,
                      n_std=1,
                      show_ylabel=True,
                      maxlen=0,
                      plot_ax=None,
                      show_xlabel=True,
                      first=True,
                      plot_noise=None,
                      infer_ylim=False,
                      lower_bound=None,
                      split_range=None,
                      hide=False,
                      plot_config=None,
                      custom_order=None,
                      no_legend=False,
                      ):
    lowest_doe_samples = 1e10
    results = {}
    if plot_ax is None:
        fig, ax = plt.subplots(figsize=(25, 16))
    else:
        ax = plot_ax
    ax_max = -np.inf
    ax_min = np.inf
    if use_empirical_regret:
        emp_regrets = get_empirical_regret(
            data_dict, metric=preprocessing[0][2]['metric'])
        for step_ in preprocessing:
            if step_[0] is compute_regret:
                step_[2]['regret'].update(emp_regrets)
    else:
        emp_regrets = None

    min_ = np.inf
    data_string = {}
    benchmark_data = {}
    for run_name, files in data_dict.items():
        plot_layout = copy(PLOT_LAYOUT)

        if hide.get(run_name, False):
            plot_layout['c'] = 'none'
            plot_layout['label'] = '__nolabel__'
        else:
            plot_layout['linestyle'] = linestyle
            if run_name in MARKERS.keys():
                plot_layout['marker'] = MARKERS[run_name]
            plot_layout['c'] = COLORS.get(run_name, 'k')
            plot_layout['label'] = NAMES.get(run_name, 'Nameless Run') + ' ' + path_name
            if plot_layout['label'] == 'Nameless Run':
                continue
        if no_legend:
            plot_layout['label'] = '__nolabel__'
        result_dataframe = get_dataframe(files, preprocessing)
        # convert to array and plot

        data_array = result_dataframe.to_numpy()
        if only_plot > 0:
            data_array = data_array[:, 0:only_plot]
        data_array = data_array.astype(np.float64)
        only_complete = True
        if only_complete:
            complete_mask = ~np.any(np.isnan(data_array), axis=0)
            data_array = data_array[:, complete_mask]

        benchmark_data[run_name] = data_array

        y_mean = data_array.mean(axis=1)
        y_std = sem(data_array, axis=1)
        markevery = np.floor(len(y_mean) / n_markers).astype(int)
        plot_layout['markevery'] = markevery

        if maxlen:
            y_mean = y_mean[0:maxlen]
            y_std = y_std[0:maxlen]
            X = np.arange(0 + 1, maxlen + 1)

        else:
            X = np.arange(1, len(y_mean) + 1)

        y_max = (y_mean + y_std)[start_at:].max()
        y_min = (y_mean - y_std)[start_at:].min()

        if start_at > 0 and not infer_ylim:
            X = X[start_at:]
            y_mean = y_mean[start_at:]
            y_std = y_std[start_at:]
        if flip:
            y_mean = y_mean + 1

        if percentage:
            y_mean = y_mean * 100
            y_std = y_std * 100
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        ax.plot(X, y_mean, **plot_layout)
        #ax.plot(X, data_array, **plot_layout)
        ax.fill_between(X, y_mean - n_std * y_std, y_mean + n_std
                        * y_std, alpha=0.1, color=plot_layout['c'])
        ax.plot(X, y_mean - n_std * y_std, alpha=0.5, color=plot_layout['c'])
        ax.plot(X, y_mean + n_std * y_std, alpha=0.5, color=plot_layout['c'])
        min_ = min((y_mean - n_std * y_std).min(), min_)

        ax_max = np.max([y_max, ax_max])
        ax_min = np.min([y_min, ax_min])

        data_string[run_name] = str(np.round(y_mean[-1], 2)) + \
            u"\u00B1" + str(np.round(y_std[-1], 2))

    if fix_range is not None:
        ax.set_ylim(fix_range)
    elif infer_ylim:
        diff_frac = (ax_max - ax_min) / 20
        ax.set_ylim([ax_min - diff_frac, ax_max + diff_frac])

    ax.axvline(x=init, color='k', linestyle=':', linewidth=2)
    if not show_xticks:
        ax.tick_params(axis='x', labelsize=0)

    else:
        ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if plot_noise is not None:
        if first:
            label = '$\sigma_\\varepsilon$'
        else:
            label = '__nolabel__'
        ax.axhline(np.log10(plot_noise), c='orange', label=label, linewidth=2)
    if lower_bound is not None:
        ax.axhline(y=lower_bound, color='k', linestyle='--', linewidth=2)

    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=plot_config['fontsizes']['metric'])
    ax.set_title(title, fontsize=plot_config['fontsizes']['benchmark_title'])
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=plot_config['fontsizes']['metric'])
    if first:
        print('First')
        handles, labels = ax.get_legend_handles_labels()
        sorted_indices = np.argsort(labels[:-1])
        sorted_indices = np.append(sorted_indices, len(labels) - 1)
        if custom_order is not None:
            try:
                ax.legend(np.array(handles)[custom_order],
                          np.array(labels)[custom_order], fontsize=plot_config['fontsizes']['legend'] * 1.3)
            except:
                pass
        else:
            ax.legend(np.array(handles)[sorted_indices],
                      np.array(labels)[sorted_indices], fontsize=plot_config['fontsizes']['legend'] * 1.3)

    ax.grid(visible=True)
    return benchmark_data, data_string


if __name__ == '__main__':

    acqs = [
        # 'ScoreBO_J_HR',
        # 'ScoreBO_J_IG',
        # 'GIBBON',
        # 'JES-e-LB2',
        # 'NEI',
        # 'PES',
        # 'NEI_MAP_AL_MAP',
        # 'ScoreBO_J_HR_',
        # 'NEI_wide',
        # 'NEI_correct',

        'LogNEI',
        'MES',
        'MCpi-MES',
        'MCpi-LogEI',
        'PiBO',
        'Sampling'
    ]
    presentation = True
    all_data = {}
    include_ranking = False
    config = 'pd1_colabo'
    #acqs = 'SAL_WS', 'SAL_HR_default', 'SAL_JS', 'BQBC', 'QBMGP', 'BALM', 'BALD', 'SAL_KL'
    # acqs = 'SAL_KLrev', 'SAL_KL', 'BALD', 'BALM'#, 'SAL_HR_default'
    plot_config = SIZE_CONFIGS[config]

    benchmarks = BENCHMARK_PACKS[config]['names']

    lower_bounds = BENCHMARK_PACKS[config].get('best', [None] * len(benchmarks))
    get_whatever = get_metric
    num_benchmarks = len(benchmarks)
    if num_benchmarks == 0:
        raise ValueError('No files')

    num_rows = plot_config['subplots']['nrows']
    cols = plot_config['subplots']['ncols']

    empirical_regret_flag = plot_config.get('empirical_regret', False)
    if include_ranking and plot_config['subplots']['nrows'] == 1            :
        plot_config['subplots']['ncols'] += 1
    if presentation:
        plot_config['subplots']['figsize'] = (
            plot_config['subplots']['figsize'][0] * 1.2, plot_config['subplots']['figsize'][1])
    fig, axes = plt.subplots(**plot_config['subplots'])

    paths_to_exp = [
        # 'results/20230828_norm',
        # 'results/20230903_gp_noprior',
        # 'results/20230903_noprior',
        # 'results/20230903_good_beta10',
        # 'results/20230907_good_beta20',
        # 'results/20230907_bad_beta20',
        # 'results/20230908_norm_newbench',
        # 'results/20230908_good_newbench_sq',
        # 'results/20230911_bad_default',
        # 'results/20230913_syn_good',
        # 'results/20230913_syn_bad',
        # 'results/20230913_syn_noprior',
        # 'results/20230913_syn_hm6robust',
        # 'results/20230916_syn_good',

        #'results/20230917_syn_good_poly',

        #'results/20230919_syn_norm',
        #'results/20230920_syn_norm',
        #'results/20230920_syn_good_pibo',
        #'results/20230920_syn_good_b1',
        #'results/20230921_syn_good_b1_noise0',
        #'results/20230921_syn_good_b1_noise0_pow1',
        #'results/20230925_pd1',
        #'results/20230925_pd_025',
        #'results/20230925_pd_075',
        #'results/20230925_pd_05_b2.5',
        #'results/20230925_pd_05_b2.5',
        'results/20230925_pd1_paper',
        #'results/20230921_syn_good_b5_noise0_pow1_init3',
        
        #'results/20230921_syn_good_b1_noise0_loc-2',
        #'results/20230920_syn_good_pibo'
        #'results/20230919_syn_good_poly',
        #'results/20230919_syn_bad_poly_ops1',
        #'results/20230919_syn_good_poly_ops1',

        #'results/20230916_syn_norm',
        # 'results/20230913_syn_noprior_2048n01',
        # 'results/20230913_syn_noprior_8kfeat'
        # 'results/20230913_syn_normal',
        # 'results/20230913_syn_normal',
        # 'results/20230913_syn_noprior_lscons_e-2',

        # 'results/20230908_noprior_newbench',
        # 'results/20230909_noprior_newbench',
        # 'results/20230909_bad_newbench',
        # 'results/20230907_bad_beta_newprior',
        # 'results/20230905_bad_beta30',
        # 'results/20230904_pd1',
        # 'results/20230830_piMC_goodprior_mean2',
        # 'results/20230830_piMC_badprior_mean2',
        # 'results/20230823_piMC_badprior'
        # 'results/20230823_piMC_mvprior',
        # 'results/20230823_piMC_goodprior_mv'
        # 'results/20230807_Transformer_warp',
        # 'results/20230525_ScoreBO_budget'
        # 'results/20230822_ucbcomp_qbatch'
        # 'results/20230511_ScoreBO_RFFLarge'
        # 'results/20230807_Transformer_warp_rebuttal_1&5'
        # 'results/20230801_HPO',
        # 'results/20230613_init_Grie5D'
        # 'results/20230502_al_with_outputscale',
        # 'results/20230803_al_with_outputscale',
        # 'results/20230804_al_outputscale',
        # 'results/20230805_al_outputscale',
        # 'results/20230805_Transformer_warp_rebuttal',
        # 'results/20230806_Transformer_warp',
        # 'results/20230804_Transformer_warp',
        # 'results/20230814_mcpi_q'

    ]
    plot_metric = ['Simple Regret', 'Simple Regret', 'Missclass. Rate']
    bench_len = {'gp_2_2_2dim': 60, 'gp_2_2_2_2_2dim': 99, 'lasso_dna': 150, 'pd1_wmt': 40, 'pd1_cifar': 40, 'pd1_lm1b': 40,}
                   # 'hartmann3': 125, 'hartmann4': 150, 'hartmann6': 200, 'levy5': 175}

    linestyles = ['solid', 'dashed', 'solid']
    hide_list = [False, False, False]
    hide_acq = {acq: h for acq, h in zip(acqs, hide_list)}
    hide_acq = {}
    path_name = ['', '', '', '- LogNormal Prior', '- BoTorch Prior',
                 '- Broad lognormal prior', '']

    all_files_list = [get_files_from_experiment(
        path_, benchmarks, acqs) for path_ in paths_to_exp]
    data_string_all = {}
    for path_idx, path_to_exp in enumerate(paths_to_exp):
        files = get_files_from_experiment(
            path_to_exp, benchmarks, acqs)
        files = {bm: files[bm] for bm in benchmarks}
        for benchmark_idx, (benchmark_name, paths) in enumerate(files.items()):
            if isinstance(plot_config['compute'], tuple):
                compute_type = plot_config['compute'][benchmark_idx]
            else:
                compute_type = plot_config['compute']

            if isinstance(plot_config['metric_name'], tuple):
                metric_name = plot_config['metric_name'][benchmark_idx]
            else:
                metric_name = plot_config['metric_name']

            if isinstance(plot_config['metric'], tuple):
                metric = plot_config['metric'][benchmark_idx]
            else:
                metric = plot_config['metric']

            regret = get_regret(benchmark_name)

            if num_benchmarks == 1:
                ax = axes

            elif num_rows == 1:
                ax = axes[benchmark_idx]
            else:
                print(benchmark_idx, int(benchmark_idx / cols), benchmark_idx % cols)
                ax = axes[int(benchmark_idx / cols), benchmark_idx % cols]

            preprocessing = [(plot_config['get_whatever'], [], {'metric': metric}), (compute_type, [], {
                'log': plot_config['log_regret'], 'regret': regret})]
            results = {}
            split_range = plot_config.get('split_range', {}).get(benchmark_name, None)

            print(benchmark_name)
            if split_range is not None:
                divider = make_axes_locatable(ax)
                bonus_ax = divider.append_axes(
                    "top", size=f"35%", pad=0.03, sharex=ax)

                all_data[benchmark_name], data_string_all[benchmark_name] = plot_optimization(paths,
                                                                                              xlabel='',
                                                                                              ylabel='',
                                                                                              n_std=0.5,
                                                                                              # plot_config['plot_args'].get('infer_ylim', False),
                                                                                              infer_ylim=False,  # plot_config['plot_args'].get(
                                                                                              # 'infer_ylim', False),
                                                                                              start_at=0,

                                                                                              fix_range=split_range[0],
                                                                                              flip=plot_config.get(
                                                                                                  'flip', False),
                                                                                              percentage=plot_config.get(
                                                                                                  'percentage', False),
                                                                                              preprocessing=preprocessing,
                                                                                              plot_ax=bonus_ax,
                                                                                              # benchmark_idx == len(files) - 1,
                                                                                              #
                                                                                              first=benchmark_idx == len(
                                                                                                  files) - 1,
                                                                                              n_markers=10,
                                                                                              show_ylabel=benchmark_idx % cols == 0,
                                                                                              init=plot_config.get(
                                                                                                  'init', init[benchmark_name]),  # init[benchmark_name],
                                                                                              maxlen=bench_len.get(
                                                                                                  benchmark_name, 0),
                                                                                              title=BENCHMARK_NAMES[benchmark_name],
                                                                                              use_empirical_regret=empirical_regret_flag and compute_type is compute_regret,
                                                                                              linestyle=linestyles[path_idx],
                                                                                              path_name=path_name[path_idx],
                                                                                              hide=hide_acq,
                                                                                              split_range=False,
                                                                                              no_legend=True,
                                                                                              show_xlabel=True,
                                                                                              show_xticks=False,
                                                                                              lower_bound=lower_bounds[benchmark_idx],
                                                                                              plot_config=plot_config,
                                                                                              plot_noise=BENCHMARK_PACKS[config].get(
                                                                                                  'noise', [None] * len(benchmarks))[benchmark_idx],
                                                                                              #custom_order=[0, 1, 2]
                                                                                              custom_order=[
                                                                                                  0, 2, 4, 1, 3]
                                                                                              )
                all_data[benchmark_name], data_string_all[benchmark_name] = plot_optimization(paths,
                                                                                              xlabel='Iteration',
                                                                                              ylabel=metric_name,
                                                                                              n_std=0.5,
                                                                                              # plot_config['plot_args'].get('infer_ylim', False),
                                                                                              infer_ylim=False,  # plot_config['plot_args'].get(
                                                                                              # 'infer_ylim', False),
                                                                                              start_at=plot_config['plot_args'].get(
                                                                                                  'start_at', init[benchmark_name] - 2) + 100 * int(benchmark_name == 'active_hartmann6') ,

                                                                                              fix_range=split_range[1],
                                                                                              flip=plot_config.get(
                                                                                                  'flip', False),
                                                                                              percentage=plot_config.get(
                                                                                                  'percentage', False),
                                                                                              preprocessing=preprocessing,
                                                                                              plot_ax=ax,
                                                                                              # benchmark_idx == len(files) - 1,
                                                                                              #
                                                                                              first=benchmark_idx == len(
                                                                                                  files) - 1,
                                                                                              n_markers=10,
                                                                                              show_ylabel=benchmark_idx % cols == 0,
                                                                                              init=plot_config.get(
                                                                                                  'init', init[benchmark_name]),  # init[benchmark_name],
                                                                                              maxlen=bench_len.get(
                                                                                                  benchmark_name, 0),
                                                                                              title=BENCHMARK_NAMES[benchmark_name],
                                                                                              use_empirical_regret=empirical_regret_flag and compute_type is compute_regret,
                                                                                              linestyle=linestyles[path_idx],
                                                                                              path_name=path_name[path_idx],
                                                                                              hide=hide_acq,
                                                                                              split_range=plot_config.get(
                                                                                                  'split_range', {}).get(benchmark_name, None),
                                                                                              no_legend=False,
                                                                                              show_xlabel=True,
                                                                                              lower_bound=lower_bounds[benchmark_idx],
                                                                                              plot_config=plot_config,
                                                                                              plot_noise=BENCHMARK_PACKS[config].get(
                                                                                                  'noise', [None] * len(benchmarks))[benchmark_idx],
                                                                                              #custom_order=[0, 1, 2]
                                                                                              custom_order=None
                                                                                              )
            else:
                all_data[benchmark_name], data_string_all[benchmark_name] = plot_optimization(paths,
                                                                                              xlabel='Iteration',
                                                                                              ylabel=metric_name,
                                                                                              n_std=0.5,
                                                                                              # plot_config['plot_args'].get('infer_ylim', False),
                                                                                              infer_ylim=False,  # plot_config['plot_args'].get(
                                                                                              # 'infer_ylim', False),
                                                                                              start_at=6,  # plot_config['plot_args'].get(
                                                                                              # 'start_at', init[benchmark_name]) + 100 * int(benchmark_name == 'active_hartmann6') ,

                                                                                              # fix_range=(-1, 4),
                                                                                              flip=plot_config.get(
                                                                                                  'flip', False),
                                                                                              percentage=plot_config.get(
                                                                                                  'percentage', False),
                                                                                              preprocessing=preprocessing,
                                                                                              plot_ax=ax,
                                                                                              # benchmark_idx == len(files) - 1,
                                                                                              #
                                                                                              first=benchmark_idx == len(
                                                                                                  files) - 1,
                                                                                              n_markers=10,
                                                                                              show_ylabel=benchmark_idx % cols == 0,
                                                                                              init=2,  # plot_config.get(
                                                                                              # 'init', init[benchmark_name]),  # init[benchmark_name],
                                                                                              maxlen=bench_len.get(
                                                                                                  benchmark_name, 0),
                                                                                              title=BENCHMARK_NAMES[benchmark_name],
                                                                                              use_empirical_regret=empirical_regret_flag and compute_type is compute_regret,
                                                                                              linestyle=linestyles[path_idx],
                                                                                              path_name=path_name[path_idx],
                                                                                              hide=hide_acq,
                                                                                              split_range=plot_config.get(
                                                                                                  'split_range', {}).get(benchmark_name, None),
                                                                                              no_legend=benchmark_idx != len(
                                                                                                  files) - 1,
                                                                                              show_xlabel=True,
                                                                                              lower_bound=lower_bounds[benchmark_idx],
                                                                                              plot_config=plot_config,
                                                                                              plot_noise=BENCHMARK_PACKS[config].get(
                                                                                                  'noise', [None] * len(benchmarks))[benchmark_idx],
                                                                                              custom_order=[0, 1, 2, 3, 4]
                                                                                              # custom_order=[
                                                                                              #    0, 6, 2, 3, 4, 5, 1]
                                                                                              )
    from copy import deepcopy

    # breakpoint()
    data_copy = deepcopy(all_data)
    separate_ranking = True

    # fig.suptitle('Synthetic', fontsize=36)

    # compute_significance(plot_config['reference_run'], data_copy)

    if include_ranking:
        try:
            a = axes[-1][-1]
        except:
            a = axes[-1]
        if separate_ranking:
            plt.show()
            fig, a = plt.subplots(1, 1, figsize=(6, 6))
            compute_ranking(all_data, a, plot_config, legend=True)

        else:
            compute_ranking(all_data, a, plot_config)
            fig.legend(loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=len(acqs),
                       fontsize=plot_config['fontsizes']['legend'] + 5)
    else:
        pass
        #plt.legend(fontsize=plot_config['fontsizes']['legend'] + 1)
    plt.tight_layout(**plot_config['tight_layout'])
    plt.savefig(f'neurips_plots/{config}_good.pdf', bbox_inches='tight')
    plt.show()
