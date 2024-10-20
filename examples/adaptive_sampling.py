import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from mcmc_samplers.adaptive_sampler import AdaptiveMetropolisHastingsSampler
from mcmc_samplers.visualization import SamplerVisualizer
import numpy as np
# Define the target distribution
def target_distribution(x):
    return 0.3 * np.exp(-0.5 * ((x - 2) / 0.5) ** 2) + \
           0.7 * np.exp(-0.5 * ((x + 2) / 0.5) ** 2)
# Set parameters
initial_state = 0.0
num_samples = 5000
burn_in = 1000

# Instantiate the adaptive sampler
adaptive_mh_sampler = AdaptiveMetropolisHastingsSampler(
    target_distribution=target_distribution,
    initial_proposal_std=1.0,
    adaptation_interval=500
)

# Run sampling
samples = adaptive_mh_sampler.sample(
    initial_state=initial_state,
    num_samples=num_samples,
    burn_in=burn_in
)

print(f'Acceptance Rate: {adaptive_mh_sampler.acceptance_rate:.2f}')

# Visualize the results
SamplerVisualizer.plot_distribution(
    samples=adaptive_mh_sampler.samples,
    target_distribution=target_distribution,
    title='Adaptive Metropolis-Hastings Sampling'
)
