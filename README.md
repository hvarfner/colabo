# Official repository for the ColaBO ICLR 2024 submission

This repository contains all the necessary code and run scripts to reproduce the results in the paper and Appendix, including the priors used for each experiment. All baselines and ColaBO are implemented in BoTorch, making use of a PathPosterior object, which is the MC-approximated posterior.


### TO RUN:
0. Retrieve the necessary repositories:     
```git clone **repo_name**```
1. Install the environment:   ```conda env create -f env.yaml # This can cause an error with mf-prior-bench (JAHS-bench specifically), but can safely be ignored```
 (python 3.9 is the only version tested for support)
3. Add the necessary paths to PYTHONPATH:     
```source add_paths.sh```
4. ```cd botorch``` 

##### QUICK START
To simply run ColaBO (with noise-adapted LogEI) on Hartmann4, run

```python benchmarking/ax_run.py```

and the output can be found in results/test. 

```True Eval``` = The (noiseless!) value on the benchmark, used to calculate simple regret.


###### Algorithm Options

```algorithm: colabo_logei, colabo_mes, logei, mes, pibo, sampling```. 

In order to run ColaBO with/without a prior, additionally enter ```algorithm.use_prior=[True, False]``` in the command line.
```++algorithm.acq_kwargs.estimate=MC```

Additional options can be found in each algorithm's .yaml-file.

###### Benchmark Options
The ```noise_std, num_iters, num_init``` of each benchmark can also be altered by entering ```benchmark.example_argument = value```, such as

```benchmark=hartmann6 benchmark.noise_std = 0.0 benchmark.num_init=10 benchmark.num_iters=40``` to run noiseless Hartmann-6 for 40 iterations with 10 initial DoE samples + the remaining 30 samples as BO. *The default settings for each benchmark are already in each config, so this needs not to be changed to reproduce*.

Synthetic benchmark options: ```hartmann3, hartmann4, hartmann6, levy5, rosenbrock6, stybtang7```
The complete list of arguments is available in configs/algorithm. Alternatively, the full list of options (```seed```, ```experiment_group``` etc.) appears when entering an incorrect options argument.

##### PD1 benchmarks.
The PD1 benchmarks require the download of datasets from the mf-prior-bench repository. The extensive how-to is available in that repo's (https://github.com/automl/mf-prior-bench) readme, but in short:

Go into the mf-prior-bench repository, and download the required datasets for PD1:
```python -m mfpbench.download```

After doing so, ensure that the surrogates (two JSON-files per benchmark) can be found in mf-prior-bench/data/surrogates. If you run into issues, we kindly refer you to the README of the mf-prior-bench repository.
The name of the PD1 tasks are ```wmt, lm1b, cifar```

