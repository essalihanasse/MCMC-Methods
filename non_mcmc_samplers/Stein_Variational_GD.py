import numpy as np
import matplotlib.pyplot as plt

class SteinVariationalGradientDescent:
    def __init__(self, U, grad_U, kernel_func, epsilon, num_particles):
        """
        Initialize the Stein Variational Gradient Descent sampler.

        Parameters:
        - U: Function that computes the potential energy (negative log probability density).
        - grad_U: Function that computes the gradient of U.
        - kernel_func: Function that computes the kernel and its gradients.
        - epsilon: Step size for updates.
        - num_particles: Number of particles to use in the approximation.
        """
        self.U = U
        self.grad_U = grad_U
        self.kernel_func = kernel_func
        self.epsilon = epsilon
        self.num_particles = num_particles

    def update(self, particles):
        """
        Perform one SVGD update step on the particles.

        Parameters:
        - particles: Current particles, a NumPy array of shape (num_particles, dimension).

        Returns:
        - updated_particles: Updated particles after one SVGD step.
        """
        # Compute kernel and gradients
        kernel_matrix, grad_kernel = self.kernel_func(particles)
        grad_log_p = np.array([self.grad_U(p) for p in particles])

        phi = (kernel_matrix @ grad_log_p + np.sum(grad_kernel, axis=1)) / self.num_particles

        # Update particles
        updated_particles = particles + self.epsilon * phi
        return updated_particles

    def sample(self, initial_particles, num_iterations, burn_in=0):
        """
        Generate samples using SVGD.

        Parameters:
        - initial_particles: Initial particles, a NumPy array of shape (num_particles, dimension).
        - num_iterations: Number of SVGD iterations to perform.
        - burn_in: Number of initial iterations to discard.

        Returns:
        - samples: NumPy array of particles after burn-in.
        """
        particles = initial_particles.copy()
        samples = []

        for i in range(num_iterations):
            particles = self.update(particles)
            if i >= burn_in:
                samples.append(particles.copy())

            if (i + 1) % 100 == 0 or i == num_iterations - 1:
                v_values = particles[:, 0]
                print(f"Iteration {i + 1}/{num_iterations}, v min: {v_values.min()}, v max: {v_values.max()}")

        return np.array(samples)

# Neal's Funnel distribution parameters
sigma_v = 3.0  # Standard deviation for v
D = 2          # Dimensionality of x

def safe_exp_neg_v(v):
    """
    Safely compute exp(-v) by capping it to prevent numerical overflow.
    """
    # Cap v to prevent overflow in exp(-v)
    if v < -20:
        return np.exp(20)  # Approximate exp(-v) when v is very negative
    else:
        return np.exp(-v)

def U(q):
    v = q[0]
    x = q[1:]
    exp_neg_v = safe_exp_neg_v(v)
    term1 = 0.5 * (v / sigma_v) ** 2
    term2 = 0.5 * np.sum(x ** 2) * exp_neg_v
    return term1 + term2

def grad_U(q):
    v = q[0]
    x = q[1:]
    exp_neg_v = safe_exp_neg_v(v)
    # Gradient w.r.t v
    dU_dv = (v / sigma_v ** 2) - 0.5 * np.sum(x ** 2) * exp_neg_v
    # Gradient w.r.t x_i
    dU_dx = x * exp_neg_v
    return np.concatenate(([dU_dv], dU_dx))

# RBF Kernel function and its gradient
def kernel_func(particles):
    pairwise_diffs = particles[:, np.newaxis, :] - particles[np.newaxis, :, :]
    sq_distances = np.sum(pairwise_diffs ** 2, axis=2)
    # Ensure h is not too small
    median_sq_dist = np.median(sq_distances)
    h = median_sq_dist / np.log(particles.shape[0] + 1)
    h = max(h, 1e-3)  # prevent h from being too small
    kernel_matrix = np.exp(-sq_distances / h)
    grad_kernel = -2 * pairwise_diffs / h * kernel_matrix[:, :, np.newaxis]
    return kernel_matrix, grad_kernel

# Initialize SVGD parameters
epsilon = 0.001      # Reduced step size for stability
num_particles = 300  # Number of particles
num_iterations = 1000
burn_in = 200

# Initial particles with smaller variance in v
np.random.seed(42)
initial_v = np.random.normal(0, 1.0, size=(num_particles, 1))
initial_x = np.random.normal(0, 1, size=(num_particles, D))
initial_particles = np.hstack((initial_v, initial_x))

# Create SVGD instance
svgd = SteinVariationalGradientDescent(U, grad_U, kernel_func, epsilon, num_particles)

# Run SVGD to get samples
samples = svgd.sample(initial_particles, num_iterations, burn_in)

# Reshape samples
samples = samples.reshape(-1, D + 1)
v_samples = samples[:, 0]
x_samples = samples[:, 1:]

# Filter out any NaN or infinite values
valid_indices = np.isfinite(v_samples)
v_samples = v_samples[valid_indices]
x_samples = x_samples[valid_indices]

# Plot the samples
plt.figure(figsize=(8, 6))
plt.plot(v_samples, x_samples[:, 0], 'o', alpha=0.3, markersize=2)
plt.title("Samples from Neal's Funnel Distribution using SVGD")
plt.xlabel('v')
plt.ylabel('x[0]')
plt.grid(True)
plt.show()

# Plot the histogram of v
plt.figure(figsize=(8, 5))
plt.hist(v_samples, bins=50, density=True, alpha=0.7, color='green')
plt.title('Histogram of v')
plt.xlabel('v')
plt.ylabel('Density')
plt.grid(True)
plt.show()
