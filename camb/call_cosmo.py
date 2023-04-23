import os
import sys

import subprocess
import numpy as np

DIRPATH = os.path.dirname(os.path.abspath(__file__))

def custom_round(x):
    #breakpoint()
    #x = [str(x_)[0:5] for x_ in x]
    return x


def cosmo(x: np.ndarray):
    
    with open(f'{DIRPATH}/CAMBfeb09/params.ini', 'r') as file:
        data = file.readlines()

    data[34] = f'ombh2={x[0]}\n'
    data[35] = f'omch2={x[1]}\n'
    data[36] = f'omnuh2={0.0}\n'
    data[37] = f'omk={x[3-1]}\n'
    data[38] = f'hubble={x[4-1]}\n'
    data[51] = f'temp_cmb={x[5-1]}\n'
    data[52] = f'helium_fraction={x[6-1]}\n'
    data[53] = f'massless_neutrinos={x[7-1]}\n'
    data[69] = f'scalar_amp(1)={x[8-1]}\n'
    data[70] = f'scalar_spectral_index(1)={x[9-1]}\n'
    data[93] = f'RECFAST_fudge={x[10-1]}\n'
    data[94] = f'RECFAST_fudge_He={x[11-1]}\n'

    with open(f'{DIRPATH}/CAMBfeb09/params.ini', 'w') as file:
        file.writelines(data)

    subprocess.run(
        [f"cd {DIRPATH}/CAMBfeb09 && ./camb params.ini"], shell=True)
    res = subprocess.run(
        [f"cd {DIRPATH}/lrgdr7like && ./getlrgdr7like"], shell=True, capture_output=True)
    return float(res.stdout.strip())


def defaults():
    defs = ([
        0.0225740,
        0.116197,
        0,
        69.0167,
        2.726,
        0.24,
        3.04,
        2.15547e-9,
        0.959959,
        1.14,
        0.86
    ])
    return defs


def bounds():
    bds = ([
        [0.01, 0.05],
        [0.01, 0.15],
        [0.01, 0.05],
        [60, 80],
        [2.7, 2.8],
        [0.2, 0.3],
        [2.9, 3.09],
        [1.5e-9, 2.6e-8],
        [0.72, 2],
        [0, 3],
        [0, 3]
    ])
    return bds


if __name__ == "__main__":
    defs = defaults()
    print(cosmo(np.array(defs)))
