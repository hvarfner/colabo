
# coding: utf-8

# ## High-Dimensional sample-efficient Bayesian Optimization with SAASBO
# 
# This tutorial shows how to use the Sparse Axis-Aligned Subspace Bayesian Optimization (SAASBO) 
# method for high-dimensional Bayesian optimization [1]. SAASBO places strong priors on the 
# inverse lengthscales to avoid overfitting in high-dimensional spaces. Specifically, SAASBO 
# uses a hierarchical sparsity prior consisting of a global shrinkage parameter 
# $\tau \sim \mathcal{HC}(\beta)$ and inverse lengthscales $\rho_d \sim \mathcal{HC}(\tau)$ 
# for $d=1, \ldots, D$, where $\mathcal{HC}$ is the half-Cauchy distribution. 
# While half-Cauchy priors favor values near zero they also have heavy tails, which allows the 
# inverse lengthscales of the most important parameters to escape zero. To perform inference in the 
# SAAS model we use Hamiltonian Monte Carlo (HMC) as we found that to outperform MAP inference.
# 
# We find that SAASBO performs well on problems with hundreds of dimensions. As we rely on HMC 
# and in particular the No-U-Turn-Sampler (NUTS) for inference, the overhead of SAASBO scales 
# cubically with the number of datapoints. Depending on the problem, using more than a few hundred
# evaluations may not be feasible as SAASBO is designed for problems with a limited evaluation budget.
# 
# In general, we recommend using [Ax](https://ax.dev) for a simple BO setup like this one. See [here](https://ax.dev/tutorials/saasbo.html) for a SAASBO tutorial in Ax, which uses the Noisy Expected Improvement acquisition function. To customize the acquisition function used with SAASBO in Ax, see the [custom acquisition tutorial](./custom_acquisition), where adding `\"surrogate\": Surrogate(SaasFullyBayesianSingleTaskGP),` to the `model_kwargs` of `BOTORCH_MODULAR` step is sufficient to enable the SAAS model.
# 
# [1]: [D. Eriksson, M. Jankowiak. High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces. Proceedings of the Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, 2021.](https://proceedings.mlr.press/v161/eriksson21a.html)

# In[1]:


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from torch.quasirandom import SobolEngine


SMOKE_TEST = os.environ.get("SMOKE_TEST")


# In[2]:


tkwargs = {
    "device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}


# The time to fit the SAAS model can be decreased by lowering
# `WARMUP_STEPS` and `NUM_SAMPLES`. We recommend using 512 warmup steps and 256 samples when
# possible and to not use fewer than 256 warmup steps and 128 samples. By default, we only
# keep each 16th sample which with 256 samples results in 32 hyperparameter samples.

# In[3]:

SMOKE_TEST = True
WARMUP_STEPS = 512 if not SMOKE_TEST else 32
NUM_SAMPLES = 256 if not SMOKE_TEST else 48
THINNING = 16


# ## Simple model fitting
# We generate a simple function that only depends on the first parameter and show that the SAAS
# model sets all other lengthscales to large values.

# In[4]:


train_X = torch.rand(10, 4, **tkwargs)
test_X = torch.rand(5, 4, **tkwargs)
train_Y = torch.sin(train_X[:, :1])
test_Y = torch.sin(test_X[:, :1])


# By default, we infer the unknown noise variance in the data. You can also pass in a known 
# noise variance (`train_Yvar`) for each observation, which may be useful in cases where you for example
# know that the problem is noise-free and can then set the noise variance to a small value such as `1e-6`.
# 
# In this case you can construct a model as follows:
# ```
# gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y, train_Yvar=torch.full_like(train_Y, 1e-6))
# ```

# In[5]:


gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_fully_bayesian_model_nuts(
    gp,
    warmup_steps=WARMUP_STEPS,
    num_samples=NUM_SAMPLES,
    thinning=THINNING,
    disable_progbar=False,
)
with torch.no_grad():
    print()
    posterior = gp.posterior(test_X.unsqueeze(-2).unsqueeze(-2))


# Computing the median lengthscales over the MCMC dimensions makes it clear that the first feature has the smallest lengthscale
# 

# In[13]:



cond_X = torch.rand(2, 4, **tkwargs)
cond_Y = torch.sin(cond_X[:, :1])
cond_Y
gp.condition_on_observations(cond_X, cond_Y)


# In[ ]:


import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch import fit_fully_bayesian_model_nuts
from botorch.acquisition import qExpectedImprovement
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.models.transforms import Standardize
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
from torch.quasirandom import SobolEngine

SMOKE_TEST =True
tkwargs = {"device": torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), "dtype": torch.double}

WARMUP_STEPS = 512 if not SMOKE_TEST else 32
NUM_SAMPLES = 256 if not SMOKE_TEST else 16
THINNING = 16

train_X = torch.rand(10, 4, **tkwargs)
test_X = torch.rand(5, 4, **tkwargs)
train_Y = torch.sin(train_X[:, :1])
test_Y = torch.sin(test_X[:, :1])

gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
fit_fully_bayesian_model_nuts(
    gp, warmup_steps=WARMUP_STEPS, num_samples=NUM_SAMPLES, thinning=THINNING, disable_progbar=False
)
with torch.no_grad():
    posterior = gp.posterior(test_X)

# Shape check
print(gp.covar_module.base_kernel.lengthscale.shape) # 16 x 1 x 4

cond_X = torch.rand(5, 4, **tkwargs) # 5 x 4
cond_Y = torch.sin(cond_X[:, :1]) # 5 x 1

# RuntimeError
gp.condition_on_observations(cond_X, cond_Y)


# ### Make predictions with the model
# 
# In the next cell we show how to make predictions with the SAAS model. You compute the mean
# and variance for test points just like for any other BoTorch posteriors. Note that the mean 
# and posterior will have an extra batch dimension at -3 that corresponds to the number of MCMC
# samples (which is 16 in this tutorial).

# In[33]:


print(posterior.mean.shape)
print(posterior.variance.shape)


# We also provide several convenience methods for computing different statistics over the MCMC samples:
# ```
# mixture_mean = posterior.mixture_mean
# mixture_variance = posterior.mixture_variance
# mixture_quantile = posterior.quantile(q=0.95)
# ```

# In[5]:


print(f"Ground truth:     {test_Y.squeeze(-1)}")
print(f"Mixture mean:     {posterior.mixture_mean.squeeze(-1)}")


# ## Optimize Branin embedded in a 50D space
# We take the standard 2D Branin problem and embed it in a 50D space. In particular,
# we let dimensions 0 and 1 correspond to the true dimensions. We will show that
# SAASBO is able to identify the important dimensions and efficiently optimize this function.
# We work with the domain $[0, 1]^d$ and unnormalize the inputs to the true domain of Branin 
# before evaluating the function.

# In[19]:


from botorch.test_functions import Hartmann
branin = Hartmann(dim=6, noise_std=0.1).to(**tkwargs)


def branin_emb(x):
    """x is assumed to be in [0, 1]^d"""
    lb, ub = branin.bounds
    return branin(lb + (ub - lb) * x[..., :branin.dim])


# In[20]:


DIM = 25 if not SMOKE_TEST else 10

# Evaluation budget
N_INIT = 7
N_ITERATIONS = 43 if not SMOKE_TEST else 1
BATCH_SIZE = 1 if not SMOKE_TEST else 1
print(f"Using a total of {N_INIT + BATCH_SIZE * N_ITERATIONS} function evaluations")


# ### Run the optimization
# We use 10 initial Sobol points followed by 10 iterations of BO using a batch size of 4, 
# which results in a total of 50 function evaluations. As our goal is to minimize Branin, we flip
# the sign of the function values before fitting the SAAS model as the BoTorch acquisition
# functions assume maximization.

# In[ ]:


X = SobolEngine(dimension=DIM, scramble=True, seed=0).draw(N_INIT).to(**tkwargs)
Y = branin_emb(X).unsqueeze(-1)
print(f"Best initial point: {Y.min().item():.3f}")
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
for i in range(N_ITERATIONS):
    train_Y = -1 * Y  # Flip the sign since we want to minimize f(x)
    gp = SaasFullyBayesianSingleTaskGP(
        train_X=X,
        train_Y=train_Y,
        train_Yvar=torch.ones_like(train_Y) * 1e-6,
        outcome_transform=Standardize(m=1),
    )
    fit_fully_bayesian_model_nuts(
        gp,
        warmup_steps=WARMUP_STEPS,
        num_samples=NUM_SAMPLES,
        thinning=THINNING,
        disable_progbar=True,
    )

    EI = qNoisyExpectedImprovement(model=gp, X_baseline=X)
    candidates, acq_values = optimize_acqf(
        EI,
        bounds=torch.cat((torch.zeros(1, DIM), torch.ones(1, DIM))).to(**tkwargs),
        q=BATCH_SIZE,
        num_restarts=10,
        raw_samples=1024,
    )

    Y_next = torch.cat([branin_emb(x).unsqueeze(-1) for x in candidates]).unsqueeze(-1)
    ind_best = Y_next.argmin()
    x0 = candidates[ind_best, :branin.dim].tolist()
    print(
        f"{i + 1}) New best: {Y_next[ind_best].item():.3f} @ "
        f"{x0}")
    X = torch.cat((X, candidates))
    Y = torch.cat((Y, Y_next))


# ## Plot the results
# 
# We can see that we were able to get close to the global optimium of $\approx 0.398$ after 50 function evaluations.
# 

# In[12]:


Y_np = Y.cpu().numpy()
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(np.minimum.accumulate(Y_np), color="b", label="SAASBO")
ax.plot([0, len(Y_np)], [-3.32, -3.32], "--", c="g", lw=3, label="Optimal value")
ax.grid(True)
ax.set_title(f"Branin, D = {DIM}", fontsize=20)
ax.set_xlabel("Number of evaluations", fontsize=20)
ax.set_xlim([0, len(Y_np)])
ax.set_ylabel("Best value found", fontsize=20)
ax.set_ylim([0, 8])
ax.legend(fontsize=18)
plt.show()


# ## Predict on some test points
# We fit a model using the 50 datapoints collected by SAASBO and predict on 50 test 
# points in order to see how well the SAAS model predicts out-of-sample.
# The plot shows the mean and a 95% confidence interval for each test point.

# In[14]:


train_X = SobolEngine(dimension=DIM, scramble=True, seed=0).draw(50).to(**tkwargs)
test_X = SobolEngine(dimension=DIM, scramble=True, seed=1).draw(50).to(**tkwargs)
train_Y = branin_emb(train_X).unsqueeze(-1)
test_Y = branin_emb(test_X).unsqueeze(-1)

gp = SaasFullyBayesianSingleTaskGP(
    train_X=train_X,
    train_Y=train_Y,
    train_Yvar=torch.full_like(train_Y, 1e-6),
    outcome_transform=Standardize(m=1),
)
fit_fully_bayesian_model_nuts(
    gp,
    warmup_steps=WARMUP_STEPS,
    num_samples=NUM_SAMPLES,
    thinning=THINNING,
    disable_progbar=True,
)


# In[42]:


with torch.no_grad():
    posterior = gp.posterior(test_X)
median = posterior.quantile(value=torch.tensor([0.5], **tkwargs))
q1 = posterior.quantile(value=torch.tensor([0.025], **tkwargs))
q2 = posterior.quantile(value=torch.tensor([0.975], **tkwargs))


# In[43]:


fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.plot([0, 80], [0, 80], "b--", lw=2)

yerr1, yerr2 = median - q1, q2 - median
yerr = torch.cat((yerr1.unsqueeze(0), yerr2.unsqueeze(0)), dim=0).squeeze(-1)
markers, caps, bars = ax.errorbar(
    test_Y.squeeze(-1).cpu(),
    median.squeeze(-1).cpu(),
    yerr=yerr.cpu(),
    fmt=".",
    capsize=4,
    elinewidth=2.0,
    ms=14,
    c="k",
    ecolor="gray",
)
ax.set_xlim([0, 80])
ax.set_ylim([0, 80])
[bar.set_alpha(0.8) for bar in bars]
[cap.set_alpha(0.8) for cap in caps]
ax.set_xlabel("True value", fontsize=20)
ax.set_ylabel("Predicted value", fontsize=20)
ax.set_aspect("equal")
ax.grid(True)


# ## Look a the lengthscales from the final model
# 
# As SAASBO places strong priors on the inverse lengthscales, we only expect parameters 
# 0 and 1 to be identified as important by the model since the other parameters have no effect.
# We can confirm that this is the case below as the lengthscales of parameters 0 and 1 are 
# small with all other lengthscales being large.

# In[15]:


median_lengthscales = gp.median_lengthscale
for i in median_lengthscales.argsort()[:10]:
    print(f"Parameter {i:2}) Median lengthscale = {median_lengthscales[i].item():.2e}")

