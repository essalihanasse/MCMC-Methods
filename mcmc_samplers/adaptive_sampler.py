import numpy as np
from scipy.stats import gaussian_kde

class AdaptiveMetropolisHastingsSampler:
    def __init__(self, target_distribution, initial_proposal_std=1.0, adaptation_interval=100):
        """
        Initialize the Adaptive Metropolis-Hastings sampler.

        Parameters:
        - target_distribution: Function that computes the (unnormalized) target distribution density.
        - initial_proposal_std: Initial standard deviation for the proposal distribution.
        - adaptation_interval: Number of iterations between adaptations of the proposal distribution.
        """
        self.target_distribution = target_distribution
        self.initial_proposal_std = initial_proposal_std
        self.adaptation_interval = adaptation_interval
        self.samples = []
        self.acceptance_rate = 0.0

    def sample(self, initial_state, num_samples, burn_in=0):
        samples = []
        current_state = initial_state
        acceptances = 0
        total_iterations = num_samples + burn_in
        all_samples = [current_state]
        proposal_std = self.initial_proposal_std

        for i in range(total_iterations):
            # Adapt the proposal distribution at specified intervals
            if i > 0 and i % self.adaptation_interval == 0 and len(all_samples) > 1:
                # Use KDE to estimate the distribution and adjust proposal_std
                kde = gaussian_kde(all_samples)
                proposal_std = np.sqrt(kde.covariance[0, 0])

            # Propose a new state
            proposed_state = np.random.normal(current_state, proposal_std)

            # Calculate acceptance probability
            p_current = self.target_distribution(current_state)
            p_proposed = self.target_distribution(proposed_state)
            acceptance_ratio = p_proposed / p_current
            acceptance_probability = min(1, acceptance_ratio)

            # Accept or reject the proposed state
            if np.random.rand() < acceptance_probability:
                current_state = proposed_state
                if i >= burn_in:
                    acceptances += 1

            all_samples.append(current_state)

            if i >= burn_in:
                samples.append(current_state)

        self.samples = np.array(samples)
        self.acceptance_rate = acceptances / num_samples
        return self.samples
