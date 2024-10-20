import numpy as np
from .base_sampler import MetropolisHastingsSampler

class KernelMetropolisHastingsSampler(MetropolisHastingsSampler):
    def __init__(self, target_distribution, proposal_distribution, proposal_pdf, proposal_params=None):
        """
        Initialize the Kernel-based Metropolis-Hastings sampler.

        Parameters:
        - target_distribution: Function that computes the (unnormalized) target distribution density.
        - proposal_distribution: Function that generates a proposal given the current state.
        - proposal_pdf: Function that computes the proposal distribution density.
        - proposal_params: Additional parameters for the proposal distribution.
        """
        super().__init__(target_distribution, proposal_distribution, proposal_params)
        self.proposal_pdf = proposal_pdf

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

            # Compute proposal probabilities
            q_current_given_proposed = self.proposal_pdf(current_state, proposed_state, **self.proposal_params)
            q_proposed_given_current = self.proposal_pdf(proposed_state, current_state, **self.proposal_params)

            acceptance_ratio = (p_proposed * q_current_given_proposed) / (p_current * q_proposed_given_current)
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
