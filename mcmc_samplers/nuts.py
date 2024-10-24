import numpy as np
import matplotlib.pyplot as plt

"""
This code implements the No-U-Turn Sampler (NUTS) algorithms as described in
the paper:
"The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo"
by Hoffman and Gelman (2011).

It includes:
- Naive NUTS (Algorithm 2)
- Efficient NUTS (Algorithm 3)
- NUTS with Dual Averaging (Algorithm 6)

The code runs all three samplers on the same target distribution (a standard normal distribution)
and plots their sample histograms side by side for comparison.
"""

class NaiveNuts:
    def __init__(self, epsilon, theta_0, U, grad_U, M, delta_max=1000):
        """
        Initialize the Naive No-U-Turn Sampler.

        Parameters:
        - epsilon: Step size for the leapfrog integrator.
        - theta_0: Initial position (numpy array).
        - U: Function to compute the potential energy at a given position.
        - grad_U: Function to compute the gradient of U at a given position.
        - M: Number of samples to generate.
        - delta_max: Maximum change in the Hamiltonian allowed (default is 1000).
        """
        self.epsilon = epsilon
        self.theta_0 = theta_0
        self.U = U
        self.grad_U = grad_U
        self.M = M
        self.delta_max = delta_max
        self.samples = []  # To store the samples

    def leapfrog(self, theta, r, v):
        """
        Perform a leapfrog step.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - v: Direction (+1 or -1).

        Returns:
        - theta_new: Updated position.
        - r_new: Updated momentum.
        """
        epsilon = v * self.epsilon
        r_new = r - 0.5 * epsilon * self.grad_U(theta)
        theta_new = theta + epsilon * r_new  # Assuming M^{-1} = I
        r_new = r_new - 0.5 * epsilon * self.grad_U(theta_new)
        return theta_new, r_new

    def build_tree(self, theta, r, log_u, v, j):
        """
        Recursively build the tree for the NUTS algorithm.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - log_u: Log of the slice variable u.
        - v: Direction (+1 or -1).
        - j: Current depth of the tree.

        Returns:
        - theta_minus: Leftmost position in the tree.
        - r_minus: Momentum at theta_minus.
        - theta_plus: Rightmost position in the tree.
        - r_plus: Momentum at theta_plus.
        - C_prime: Set of candidate positions and momenta.
        - s_prime: Indicator of whether to keep sampling.
        """
        if j == 0:
            # Base case: Take one leapfrog step in the direction v
            theta_prime, r_prime = self.leapfrog(theta, r, v)
            joint = self.U(theta_prime) + 0.5 * np.dot(r_prime, r_prime)
            log_joint = -joint
            if log_joint >= log_u:
                C_prime = [(theta_prime.copy(), r_prime.copy())]
            else:
                C_prime = []
            s_prime = int(log_joint >= log_u - self.delta_max)
            return theta_prime, r_prime, theta_prime, r_prime, C_prime, s_prime
        else:
            # Recursion: Build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime = self.build_tree(theta, r, log_u, v, j - 1)
            if s_prime == 0:
                return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime
            if v == -1:
                theta_minus, r_minus, _, _, C_doubleprime, s_doubleprime = self.build_tree(theta_minus, r_minus, log_u, v, j - 1)
            else:
                _, _, theta_plus, r_plus, C_doubleprime, s_doubleprime = self.build_tree(theta_plus, r_plus, log_u, v, j - 1)
            C_prime.extend(C_doubleprime)
            delta_theta = theta_plus - theta_minus
            s_prime = s_prime and s_doubleprime and (np.dot(delta_theta, r_minus) >= 0) and (np.dot(delta_theta, r_plus) >= 0)
            s_prime = int(s_prime)
            return theta_minus, r_minus, theta_plus, r_plus, C_prime, s_prime

    def sample(self):
        """
        Run the Naive NUTS sampler to generate samples.
        """
        theta_m_minus1 = self.theta_0.copy()
        self.samples.append(theta_m_minus1.copy())
        for m in range(1, self.M):
            # Resample r0 ∼ N(0, I)
            r0 = np.random.normal(0, 1, size=theta_m_minus1.shape)
            # Compute the joint log probability at the starting point
            joint = self.U(theta_m_minus1) + 0.5 * np.dot(r0, r0)
            log_joint = -joint
            # Sample log_u ∼ [−∞, log_joint]
            log_u = log_joint - np.random.exponential(1)
            # Initialize θ⁻, θ⁺, r⁻, r⁺, j, C, s
            theta_minus = theta_m_minus1.copy()
            theta_plus = theta_m_minus1.copy()
            r_minus = r0.copy()
            r_plus = r0.copy()
            j = 0
            C = [(theta_m_minus1.copy(), r0.copy())]
            s = 1

            while s == 1:
                # Choose a direction v_j ∼ Uniform({−1, 1})
                vj = np.random.choice([-1, 1])
                if vj == -1:
                    theta_minus, r_minus, _, _, C_prime, s_prime = self.build_tree(theta_minus, r_minus, log_u, vj, j)
                else:
                    _, _, theta_plus, r_plus, C_prime, s_prime = self.build_tree(theta_plus, r_plus, log_u, vj, j)
                if s_prime == 1:
                    C.extend(C_prime)
                delta_theta = theta_plus - theta_minus
                s = s_prime and (np.dot(delta_theta, r_minus) >= 0) and (np.dot(delta_theta, r_plus) >= 0)
                s = int(s)
                j += 1
            # Sample θ_m, r uniformly at random from C
            idx = np.random.choice(len(C))
            theta_m, _ = C[idx]
            self.samples.append(theta_m.copy())
            theta_m_minus1 = theta_m.copy()

class EfficientNuts:
    def __init__(self, epsilon, theta_0, U, grad_U, M, delta_max=1000):
        """
        Initialize the Efficient No-U-Turn Sampler.

        Parameters:
        - epsilon: Step size for the leapfrog integrator.
        - theta_0: Initial position (numpy array).
        - U: Function to compute the potential energy at a given position.
        - grad_U: Function to compute the gradient of U at a given position.
        - M: Number of samples to generate.
        - delta_max: Maximum change in the Hamiltonian allowed (default is 1000).
        """
        self.epsilon = epsilon
        self.theta_0 = theta_0
        self.U = U
        self.grad_U = grad_U
        self.M = M
        self.delta_max = delta_max
        self.samples = []  # To store the samples

    def leapfrog(self, theta, r, v):
        """
        Perform a leapfrog step.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - v: Direction (+1 or -1).

        Returns:
        - theta_new: Updated position.
        - r_new: Updated momentum.
        """
        epsilon = v * self.epsilon
        r_half_step = r - 0.5 * epsilon * self.grad_U(theta)
        theta_new = theta + epsilon * r_half_step  # Assuming M^{-1} = I
        r_new = r_half_step - 0.5 * epsilon * self.grad_U(theta_new)
        return theta_new, r_new

    def build_tree(self, theta, r, log_u, v, j):
        """
        Recursively build the tree for the Efficient NUTS algorithm.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - log_u: Log of the slice variable u.
        - v: Direction (+1 or -1).
        - j: Current depth of the tree.

        Returns:
        - theta_minus: Leftmost position in the tree.
        - r_minus: Momentum at theta_minus.
        - theta_plus: Rightmost position in the tree.
        - r_plus: Momentum at theta_plus.
        - theta_prime: Candidate sample from the subtree.
        - n_prime: Number of valid points in the subtree.
        - s_prime: Indicator of whether to keep sampling.
        """
        if j == 0:
            # Base case: Take one leapfrog step in the direction v
            theta_prime, r_prime = self.leapfrog(theta, r, v)
            joint = self.U(theta_prime) + 0.5 * np.dot(r_prime, r_prime)
            log_joint = -joint
            n_prime = int(log_u <= log_joint)
            s_prime = int(log_u - self.delta_max < log_joint)
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime
        else:
            # Recursion: Build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime = self.build_tree(theta, r, log_u, v, j - 1)
            if s_prime == 0:
                return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime
            if v == -1:
                theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime = self.build_tree(theta_minus, r_minus, log_u, v, j - 1)
            else:
                _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime = self.build_tree(theta_plus, r_plus, log_u, v, j - 1)
            n_total = n_prime + n_double_prime
            # Decide whether to accept theta_double_prime as theta_prime
            accept_prob = n_double_prime / max(n_total, 1)
            if np.random.uniform() < accept_prob:
                theta_prime = theta_double_prime.copy()
            s_prime = s_double_prime and (np.dot(theta_plus - theta_minus, r_minus) >= 0) and (np.dot(theta_plus - theta_minus, r_plus) >= 0)
            s_prime = int(s_prime)
            n_prime = n_total
            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime

    def sample(self):
        """
        Run the Efficient NUTS sampler to generate samples.
        """
        theta_m_minus1 = self.theta_0.copy()
        self.samples.append(theta_m_minus1.copy())
        for m in range(1, self.M):
            # Resample r0 ∼ N(0, I)
            r0 = np.random.normal(0, 1, size=theta_m_minus1.shape)
            # Compute the joint log probability at the starting point
            joint = self.U(theta_m_minus1) + 0.5 * np.dot(r0, r0)
            log_joint = -joint
            # Sample log_u ∼ [−∞, log_joint]
            log_u = log_joint - np.random.exponential(1)
            # Initialize θ⁻, θ⁺, r⁻, r⁺, j, n, s
            theta_minus = theta_m_minus1.copy()
            theta_plus = theta_m_minus1.copy()
            r_minus = r0.copy()
            r_plus = r0.copy()
            j = 0
            theta_m = theta_m_minus1.copy()
            n = 1
            s = 1

            while s == 1:
                # Choose a direction v_j ∼ Uniform({−1, 1})
                vj = np.random.choice([-1, 1])
                if vj == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime = self.build_tree(theta_minus, r_minus, log_u, vj, j)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime = self.build_tree(theta_plus, r_plus, log_u, vj, j)
                if s_prime == 1:
                    accept_prob = min(1.0, n_prime / n)
                    if np.random.uniform() < accept_prob:
                        theta_m = theta_prime.copy()
                n += n_prime
                delta_theta = theta_plus - theta_minus
                s = s_prime and (np.dot(delta_theta, r_minus) >= 0) and (np.dot(delta_theta, r_plus) >= 0)
                s = int(s)
                j += 1
            self.samples.append(theta_m.copy())
            theta_m_minus1 = theta_m.copy()

class NutsDualAveraging:
    def __init__(self, theta_0, U, grad_U, M, Madapt, delta=0.65, delta_max=1000):
        """
        Initialize the NUTS sampler with dual averaging step size adaptation.

        Parameters:
        - theta_0: Initial position (numpy array).
        - U: Function to compute the potential energy at a given position.
        - grad_U: Function to compute the gradient of U at a given position.
        - M: Total number of samples to generate.
        - Madapt: Number of adaptation steps for step size.
        - delta: Target acceptance probability (default is 0.65).
        - delta_max: Maximum change in the Hamiltonian allowed (default is 1000).
        """
        self.theta_0 = theta_0
        self.U = U
        self.grad_U = grad_U
        self.M = M
        self.Madapt = Madapt
        self.delta = delta
        self.delta_max = delta_max
        self.samples = []  # To store the samples
        self.epsilon = None  # Step size to be initialized

    def leapfrog(self, theta, r, epsilon):
        """
        Perform a leapfrog step.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - epsilon: Step size.

        Returns:
        - theta_new: Updated position.
        - r_new: Updated momentum.
        """
        r_half_step = r - 0.5 * epsilon * self.grad_U(theta)
        theta_new = theta + epsilon * r_half_step  # Assuming M^{-1} = I
        r_new = r_half_step - 0.5 * epsilon * self.grad_U(theta_new)
        return theta_new, r_new

    def find_reasonable_epsilon(self, theta):
        """
        Heuristic to find a reasonable initial value for epsilon.

        Parameters:
        - theta: Current position.

        Returns:
        - epsilon: Initial step size.
        """
        epsilon = 1.0
        r = np.random.normal(0, 1, size=theta.shape)
        joint = self.U(theta) + 0.5 * np.dot(r, r)
        theta_new, r_new = self.leapfrog(theta, r, epsilon)
        joint_new = self.U(theta_new) + 0.5 * np.dot(r_new, r_new)
        log_accept_ratio = joint - joint_new
        a = 1 if log_accept_ratio > np.log(0.5) else -1
        # Use a while loop to adjust epsilon
        while True:
            epsilon *= 2 ** a
            theta_new, r_new = self.leapfrog(theta, r, epsilon)
            joint_new = self.U(theta_new) + 0.5 * np.dot(r_new, r_new)
            log_accept_ratio = joint - joint_new
            condition = log_accept_ratio > np.log(0.5) if a == 1 else log_accept_ratio < np.log(0.5)
            if not condition:
                break
        return epsilon

    def build_tree(self, theta, r, log_u, v, j, epsilon, theta0, r0):
        """
        Recursively build the tree for the NUTS algorithm with dual averaging.

        Parameters:
        - theta: Current position.
        - r: Current momentum.
        - log_u: Log of the slice variable u.
        - v: Direction (+1 or -1).
        - j: Current depth of the tree.
        - epsilon: Step size.
        - theta0, r0: Initial position and momentum.

        Returns:
        - theta_minus: Leftmost position in the tree.
        - r_minus: Momentum at theta_minus.
        - theta_plus: Rightmost position in the tree.
        - r_plus: Momentum at theta_plus.
        - theta_prime: Candidate sample from the subtree.
        - n_prime: Number of valid points in the subtree.
        - s_prime: Indicator of whether to keep sampling.
        - alpha_prime: Sum of acceptance probabilities.
        - n_alpha_prime: Number of proposals.
        """
        if j == 0:
            # Base case: Take one leapfrog step in the direction v
            theta_prime, r_prime = self.leapfrog(theta, r, v * epsilon)
            joint = - (self.U(theta_prime) + 0.5 * np.dot(r_prime, r_prime))
            n_prime = int(log_u < joint)
            s_prime = int(log_u - self.delta_max < joint)
            # Compute delta_H
            joint0 = - (self.U(theta0) + 0.5 * np.dot(r0, r0))
            delta_joint = joint - joint0  # delta_joint = -H_new + H0
            if delta_joint >= 0:
                alpha_prime = 1.0
            else:
                # Avoid overflow in exp by capping delta_joint
                alpha_prime = np.exp(delta_joint)
            n_alpha_prime = 1
            return theta_prime, r_prime, theta_prime, r_prime, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime
        else:
            # Recursion: Build the left and right subtrees
            theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                self.build_tree(theta, r, log_u, v, j - 1, epsilon, theta0, r0)
            if s_prime == 1:
                if v == -1:
                    theta_minus, r_minus, _, _, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = \
                        self.build_tree(theta_minus, r_minus, log_u, v, j - 1, epsilon, theta0, r0)
                else:
                    _, _, theta_plus, r_plus, theta_double_prime, n_double_prime, s_double_prime, alpha_double_prime, n_alpha_double_prime = \
                        self.build_tree(theta_plus, r_plus, log_u, v, j - 1, epsilon, theta0, r0)
                # Decide whether to accept theta_double_prime as theta_prime
                accept_prob = n_double_prime / max(n_prime + n_double_prime, 1)
                if np.random.uniform() < accept_prob:
                    theta_prime = theta_double_prime.copy()
                n_prime += n_double_prime
                s_prime = s_double_prime and self.stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
                alpha_prime += alpha_double_prime
                n_alpha_prime += n_alpha_double_prime
            return theta_minus, r_minus, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime

    def stop_criterion(self, theta_minus, theta_plus, r_minus, r_plus):
        """
        Check the No-U-Turn criterion.

        Returns:
        - s: 1 if we should continue sampling, 0 otherwise.
        """
        delta_theta = theta_plus - theta_minus
        s = int((np.dot(delta_theta, r_minus) >= 0) and (np.dot(delta_theta, r_plus) >= 0))
        return s

    def sample(self):
        """
        Run the NUTS sampler with dual averaging to generate samples.
        """
        theta_m_minus1 = self.theta_0.copy()
        self.samples.append(theta_m_minus1.copy())

        # Initialize dual averaging parameters
        epsilon = self.find_reasonable_epsilon(theta_m_minus1)
        epsilon = np.clip(epsilon, 1e-5, 1e1)  # Adjusted clipping range
        self.epsilon = epsilon  # Store the initial epsilon
        mu = np.log(10 * epsilon)
        gamma = 0.05
        t0 = 10.0
        kappa = 0.75
        H_bar = 0.0
        epsilon_bar = 1.0

        for m in range(1, self.M + 1):
            # Resample r0 ∼ N(0, I)
            r0 = np.random.normal(0, 1, size=theta_m_minus1.shape)
            # Compute the joint log probability at the starting point
            joint = - (self.U(theta_m_minus1) + 0.5 * np.dot(r0, r0))
            log_joint = joint
            # Sample log_u ∼ [log_u, log_joint]
            log_u = log_joint - np.random.exponential(1)
            # Initialize θ⁻, θ⁺, r⁻, r⁺, j, n, s
            theta_minus = theta_m_minus1.copy()
            theta_plus = theta_m_minus1.copy()
            r_minus = r0.copy()
            r_plus = r0.copy()
            j = 0
            theta_m = theta_m_minus1.copy()
            n = 1
            s = 1
            alpha = 0.0
            n_alpha = 0

            while s == 1:
                # Choose a direction v_j ∼ Uniform({−1, 1})
                vj = np.random.choice([-1, 1])
                if vj == -1:
                    theta_minus, r_minus, _, _, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                        self.build_tree(theta_minus, r_minus, log_u, vj, j, epsilon, theta_m_minus1, r0)
                else:
                    _, _, theta_plus, r_plus, theta_prime, n_prime, s_prime, alpha_prime, n_alpha_prime = \
                        self.build_tree(theta_plus, r_plus, log_u, vj, j, epsilon, theta_m_minus1, r0)
                if s_prime == 1 and n_prime > 0:
                    accept_prob = min(1.0, n_prime / n)
                    if np.random.uniform() < accept_prob:
                        theta_m = theta_prime.copy()
                alpha += alpha_prime
                n_alpha += n_alpha_prime
                n += n_prime
                s = s_prime and self.stop_criterion(theta_minus, theta_plus, r_minus, r_plus)
                j += 1

            self.samples.append(theta_m.copy())
            theta_m_minus1 = theta_m.copy()

            # Dual averaging adaptation of epsilon
            if m <= self.Madapt:
                alpha_ratio = alpha / max(n_alpha, 1)
                H_bar = (1 - 1 / (m + t0)) * H_bar + (1 / (m + t0)) * (self.delta - alpha_ratio)
                log_epsilon = mu - (np.sqrt(m) / gamma) * H_bar
                epsilon = np.exp(log_epsilon)
                epsilon = np.clip(epsilon, 1e-5, 1e1)  # Adjusted clipping range
                log_epsilon_bar = m ** (-kappa) * log_epsilon + (1 - m ** (-kappa)) * np.log(epsilon_bar)
                epsilon_bar = np.exp(log_epsilon_bar)
            else:
                epsilon = epsilon_bar

        self.samples = self.samples[self.Madapt:]  # Discard adaptation samples

# Example usage
if __name__ == "__main__":
    np.random.seed(42)  # Set seed for reproducibility

    # Define the potential energy function U and its gradient
    def U(theta):
        return 0.5 * np.dot(theta, theta)

    def grad_U(theta):
        return theta

    # Parameters
    theta_0 = np.array([0.0])  # Starting position in 1D
    M_total = 15000  # Total number of samples (including adaptation for Dual Averaging NUTS)
    Madapt = 500  # Number of adaptation steps for Dual Averaging NUTS
    delta = 0.65  # Target acceptance probability for Dual Averaging NUTS
    M = M_total - Madapt  # Adjusted number of samples for other samplers

    # Create instances of all three samplers
    naive_sampler = NaiveNuts(epsilon=0.1, theta_0=theta_0, U=U, grad_U=grad_U, M=M)
    efficient_sampler = EfficientNuts(epsilon=0.1, theta_0=theta_0, U=U, grad_U=grad_U, M=M)
    dual_averaging_sampler = NutsDualAveraging(theta_0, U, grad_U, M_total, Madapt, delta=delta)

    # Run the samplers
    print("Running Naive NUTS...")
    naive_sampler.sample()
    print("Running Efficient NUTS...")
    efficient_sampler.sample()
    print("Running NUTS with Dual Averaging (Algorithm 6)...")
    dual_averaging_sampler.sample()

    # Access the samples
    naive_samples = np.array(naive_sampler.samples).flatten()
    efficient_samples = np.array(efficient_sampler.samples).flatten()
    dual_averaging_samples = np.array(dual_averaging_sampler.samples).flatten()

    # Plot the histograms of samples from all samplers
    plt.figure(figsize=(18, 5))

    # Histogram for Naive NUTS
    plt.subplot(1, 3, 1)
    plt.hist(naive_samples, bins=50, density=True, alpha=0.7, label='Naive NUTS Samples')
    x = np.linspace(-4, 4, 100)
    plt.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2), 'r-', label='Standard Normal PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Naive NUTS')

    # Histogram for Efficient NUTS
    plt.subplot(1, 3, 2)
    plt.hist(efficient_samples, bins=50, density=True, alpha=0.7, label='Efficient NUTS Samples')
    plt.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2), 'r-', label='Standard Normal PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Efficient NUTS')

    # Histogram for NUTS with Dual Averaging
    plt.subplot(1, 3, 3)
    plt.hist(dual_averaging_samples, bins=50, density=True, alpha=0.7, label='NUTS with Dual Averaging Samples')
    plt.plot(x, (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x ** 2), 'r-', label='Standard Normal PDF')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.title('NUTS with Dual Averaging (Algorithm 6)')

    plt.tight_layout()
    plt.show()
