import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

class HamiltonianMonteCarlo:
    def __init__(self, U, grad_U, epsilon, L):
        """
        Initialize the Hamiltonian Monte Carlo sampler.

        Parameters:
        - U: Function that computes the potential energy (negative log probability density).
        - grad_U: Function that computes the gradient of U.
        - epsilon: Step size for leapfrog integration.
        - L: Number of leapfrog steps per proposal.
        """
        self.U = U
        self.grad_U = grad_U
        self.epsilon = epsilon
        self.L = L

    def hamiltonian_step(self, current_q):
        """
        Perform one Hamiltonian Monte Carlo step.

        Parameters:
        - current_q: Current position.

        Returns:
        - proposed_q: Proposed new position (may be the same as current_q if rejected).
        - accepted: Boolean indicating whether the proposal was accepted.
        """
        q = current_q.copy()
        p = np.random.normal(size=len(q))
        current_p = p.copy()

        # Make a half step for momentum at the beginning
        p -= 0.5 * self.epsilon * self.grad_U(q)

        # Alternate full steps for position and momentum
        for i in range(self.L):
            # Full step for position
            q += self.epsilon * p
            # Full step for momentum, except at end of trajectory
            if i != self.L - 1:
                p -= self.epsilon * self.grad_U(q)

        # Make a half step for momentum at the end
        p -= 0.5 * self.epsilon * self.grad_U(q)
        # Negate momentum to make proposal symmetric
        p = -p

        # Compute Hamiltonian
        current_U = self.U(current_q)
        current_K = 0.5 * np.sum(current_p ** 2)
        proposed_U = self.U(q)
        proposed_K = 0.5 * np.sum(p ** 2)

        # Metropolis acceptance criterion
        delta_H = (current_U + current_K) - (proposed_U + proposed_K)
        acceptance_prob = min(1, np.exp(delta_H))
        if np.random.rand() < acceptance_prob:
            return q, True  # Accept
        else:
            return current_q, False  # Reject

    def sample(self, initial_position, num_samples, burn_in=0):
        """
        Generate samples using Hamiltonian Monte Carlo.

        Parameters:
        - initial_position: Starting point of the Markov chain (numpy array).
        - num_samples: Number of samples to generate.
        - burn_in: Number of initial samples to discard.

        Returns:
        - samples: NumPy array of samples from the target distribution.
        """
        current_q = initial_position.copy()
        samples = []
        accept_count = 0

        for i in range(num_samples + burn_in):
            q_new, accepted = self.hamiltonian_step(current_q)
            if accepted:
                accept_count += 1
            current_q = q_new
            if i >= burn_in:
                samples.append(current_q)

        acceptance_rate = accept_count / (num_samples + burn_in)
        print(f"Acceptance rate: {acceptance_rate:.2f}")

        return np.array(samples)

# Neal's Funnel distribution parameters
sigma_v = 3.0  # Standard deviation for v
D = 2          # Dimensionality of x (can be increased for higher dimensions)

def U(q):
    v = q[0]
    x = q[1:]
    term1 = 0.5 * (v / sigma_v) ** 2
    term2 = 0.5 * np.sum(x ** 2) * np.exp(-v)
    return term1 + term2

def grad_U(q):
    v = q[0]
    x = q[1:]
    # Gradient w.r.t v
    dU_dv = (v / sigma_v ** 2) - 0.5 * np.sum(x ** 2) * np.exp(-v)
    # Gradient w.r.t x_i
    dU_dx = x * np.exp(-v)
    return np.concatenate(([dU_dv], dU_dx))
# Initialize HMC parameters
epsilon = 0.01  # Step size
L = 100         # Number of leapfrog steps
hmc = HamiltonianMonteCarlo(U, grad_U, epsilon, L)

# Initial position
initial_v = 0.0
initial_x = np.zeros(D)
initial_position = np.concatenate(([initial_v], initial_x))

# Number of samples and burn-in
num_samples = 5000
burn_in = 1000

# Generate samples using HMC
samples = hmc.sample(initial_position, num_samples, burn_in)

# Extract v and x samples
v_samples = samples[:, 0]
x_samples = samples[:, 1:]

# Plot the samples
plt.figure(figsize=(8, 6))
plt.plot(v_samples, x_samples[:, 0], 'o', alpha=0.3, markersize=2)
plt.title("Samples from Neal's Funnel Distribution using HMC")
plt.xlabel('v')
plt.ylabel('x[0]')
plt.grid(True)
plt.show()

# Plot the histogram of v
plt.figure(figsize=(8, 5))
plt.hist(v_samples, bins=50, density=True, alpha=0.7, color='purple')
plt.title('Histogram of v')
plt.xlabel('v')
plt.ylabel('Density')
plt.grid(True)
plt.show()
