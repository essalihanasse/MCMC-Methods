import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from mcmc_samplers.multivariate_sampler import MultivariateMetropolisHastingsSampler
import numpy as np
import matplotlib.pyplot as plt

# Define the multivariate target distribution
def target_distribution_2d(x):
    mu = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    diff = x - mu
    exponent = -0.5 * diff.T @ np.linalg.inv(cov) @ diff
    return np.exp(exponent)

# Define the multivariate proposal distribution
def gaussian_proposal_2d(x, cov):
    return np.random.multivariate_normal(mean=x, cov=cov)

# Set parameters
initial_state = np.array([0.0, 0.0])
num_samples = 5000
burn_in = 1000
proposal_params = {'cov': np.eye(2) * 0.5}

# Instantiate and run the sampler
multivariate_sampler = MultivariateMetropolisHastingsSampler(
    target_distribution=target_distribution_2d,
    proposal_distribution=gaussian_proposal_2d,
    proposal_params=proposal_params
)

samples = multivariate_sampler.sample(
    initial_state=initial_state,
    num_samples=num_samples,
    burn_in=burn_in
)

# Plot the samples
samples_array = np.array(samples)
plt.figure(figsize=(6, 6))
plt.scatter(samples_array[:, 0], samples_array[:, 1], alpha=0.5, s=10)
plt.title('Samples from 2D Gaussian using Metropolis-Hastings')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')
plt.show()
