import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from mcmc_samplers.base_sampler import MetropolisHastingsSampler
from mcmc_samplers.visualization import SamplerVisualizer
import numpy as np

# Define the target distribution
def target_distribution(x):
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5) ** 2) + \
           0.7 * np.exp(-0.5 * ((x + 2) / 0.8) ** 2)

# Define the proposal distribution
def gaussian_proposal(x, scale):
    return np.random.normal(loc=x, scale=scale)

# Set parameters
initial_state = 0.0
num_samples = 5000
burn_in = 1000
proposal_params = {'scale': 1.0}

# Instantiate and run the sampler
mh_sampler = MetropolisHastingsSampler(
    target_distribution=target_distribution,
    proposal_distribution=gaussian_proposal,
    proposal_params=proposal_params
)

samples = mh_sampler.sample(
    initial_state=initial_state,
    num_samples=num_samples,
    burn_in=burn_in
)

print(f'Acceptance Rate: {mh_sampler.acceptance_rate:.2f}')

# Visualize the results
SamplerVisualizer.plot_distribution(
    samples=mh_sampler.samples,
    target_distribution=target_distribution,
    title='Metropolis-Hastings Sampling'
)
