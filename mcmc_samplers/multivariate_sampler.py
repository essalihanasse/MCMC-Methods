import numpy as np

class MultivariateMetropolisHastingsSampler:
    def __init__(self, target_distribution, proposal_distribution, proposal_params=None):
        """
        Initialize the Multivariate Metropolis-Hastings sampler.

        Parameters:
        - target_distribution: Function that computes the (unnormalized) target distribution density.
        - proposal_distribution: Function that generates a proposal given the current state.
        - proposal_params: Additional parameters for the proposal distribution.
        """
        self.target_distribution = target_distribution
        self.proposal_distribution = proposal_distribution
        self.proposal_params = proposal_params if proposal_params is not None else {}
        self.samples = []
        self.acceptance_rate = 0.0

    def sample(self, initial_state, num_samples, burn_in=0):
        samples = []
        current_state = initial_state
        acceptances = 0
        total_iterations = num_samples + burn_in

        for i in range(total_iterations):
            # Propose a new state
            proposed_state = self.proposal_distribution(current_state, **self.proposal_params)

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

            if i >= burn_in:
                samples.append(current_state)

        self.samples = np.array(samples)
        self.acceptance_rate = acceptances / num_samples
        return self.samples
