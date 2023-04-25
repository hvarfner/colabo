import copy
import subprocess

import numpy as np
from tqdm import tqdm

PATH = "."


def cosmo(x: np.ndarray):
    with open(f'{PATH}/CAMBfeb09/params.ini', 'r') as file:
        data = file.readlines()
    data[34] = f'ombh2={x[0]}\n'
    data[35] = f'omch2={x[1]}\n'
    data[37] = f'omk={x[2]}\n'
    data[38] = f'hubble={x[3]}\n'
    data[51] = f'temp_cmb={x[4]}\n'
    data[52] = f'helium_fraction={x[5]}\n'
    data[53] = f'massless_neutrinos={x[6]}\n'
    data[69] = f'scalar_amp(1)={x[7]}\n'
    data[70] = f'scalar_spectral_index(1)={x[8]}\n'
    data[93] = f'RECFAST_fudge={x[9]}\n'
    data[94] = f'RECFAST_fudge_He={x[10]}\n'

    with open(f'{PATH}/CAMBfeb09/params.ini', 'w') as file:
        file.writelines(data)

    try:
        out = subprocess.run(
            ["./camb", "params.ini"], cwd=f"{PATH}/CAMBfeb09", capture_output=True, timeout=600, check=True)
    except subprocess.TimeoutExpired as e:
        print("TIMEOUT", x)
        return 300
    if out.stderr or len(out.stdout) <= 3 or out.stdout[:3] != b"Age":
        print(out.stderr, out.stdout, x)
        return 300
    res = subprocess.run(["./getlrgdr7like"],
                         cwd=f"{PATH}/lrgdr7like", capture_output=True)
    try:
        return float(res.stdout.strip())
    except:
        print(res.stdout)
        print(res.stderr)
        print(x)
        # return 300


defaults = np.array([
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

bounds = np.array([
    [0.01, 0.08],
    [0.01, 0.25],
    [0.01, 0.25],
    [50, 100],
    [2.7, 2.8],
    [0.2, 0.3],
    [2.9, 3.09],
    [1.5e-9, 2.6e-8],
    [0.72, 2.7],
    [0, 100],
    [0, 100]
])

turbo_bounds = np.array([
    [0.01, 0.25],
    [0.01, 0.25],
    [0.01, 0.25],
    [50, 100],
    [2.7, 2.8],
    [0.2, 0.3],
    [2.9, 3.09],
    [1.5e-9, 2.6e-8],
    [0.72, 5],
    [0, 100],
    [0, 100]
])

steps = np.array([
    [0, 0.01],
    [0, 0.01],
    [0, 0.01],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0.1],
    [0, 1],
    [0, 1]
])


def test_bounds():
    # individual tests
    for i in range(11):
        # lower
        if steps[i, 0] > 0:
            bds = copy.deepcopy(bounds)
            while bds[i, 0] > turbo_bounds[i, 0]:
                res = cosmo(bds[:, 0].squeeze())
                print("L", i, bds[i, 0], ":", res)
                if res == 1:
                    break
                bds[i, 0] = bds[i, 0] - steps[i, 0]

        # upper
        if steps[i, 1] > 0:
            bds = copy.deepcopy(bounds)
            while bds[i, 1] < turbo_bounds[i, 1]:
                res = cosmo(bds[:, 1].squeeze())
                print("U", i, bds[i, 1], ":", res)
                if res == 300:
                    break
                bds[i, 1] = bds[i, 1] + steps[i, 1]


if __name__ == "__main__":
    print("RUNNING DEFAULT")
    print(cosmo(np.array(defaults)))
    print("TESTING BOUNDS")
    # test_bounds()
    print("TESTING RANDOM")
    for i in tqdm(range(100)):
        arr = np.random.random(
            11) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        print(cosmo(arr))
