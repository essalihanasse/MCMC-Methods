import unittest
import sys
import os

# Add the parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


from mcmc_samplers.base_sampler import MetropolisHastingsSampler
import numpy as np

class TestMetropolisHastingsSampler(unittest.TestCase):
    def test_sampler_converges_to_normal_distribution(self):
        # Define the target distribution (standard normal)
        def target_distribution(x):
            return np.exp(-0.5 * x**2)

        # Define the proposal distribution (normal)
        def proposal_distribution(x, scale):
            return np.random.normal(loc=x, scale=scale)

        # Set parameters
        initial_state = 0.0
        num_samples = 10000
        burn_in = 1000
        proposal_params = {'scale': 1.0}

        # Instantiate and run the sampler
        sampler = MetropolisHastingsSampler(
            target_distribution=target_distribution,
            proposal_distribution=proposal_distribution,
            proposal_params=proposal_params
        )

        samples = sampler.sample(
            initial_state=initial_state,
            num_samples=num_samples,
            burn_in=burn_in
        )

        # Check that the mean and standard deviation are close to 0 and 1
        mean_estimate = np.mean(samples)
        std_estimate = np.std(samples)

        self.assertAlmostEqual(mean_estimate, 0, places=1)
        self.assertAlmostEqual(std_estimate, 1, places=1)

if __name__ == '__main__':
    unittest.main()
