import numpy as np
import matplotlib.pyplot as plt

"""
This code has been inspired by:
SLICE SAMPLING
BY RADFORD M. NEAL
"""
class SliceSamplerUniDimensional:
    def __init__(self, f, x_0):
        """
        f is a function proportional to the density of x P(x)
        """
        self.f = f
        self.x_0 = x_0

    def stepping_out(self, x, y, w, m):
        u = np.random.rand()
        L = x - w * u
        R = L + w
        V = np.random.rand()
        J = int(np.floor(m * V))
        K = m - 1 - J
        while J > 0 and y < self.f(L):
            L -= w
            J -= 1
        while K > 0 and y < self.f(R):
            R += w
            K -= 1
        return L, R

    def doubling(self, x, y, w, p):
        u = np.random.rand()
        L = x - w * u
        R = L + w
        K = p
        while K > 0 and (y < self.f(L) or y < self.f(R)):
            V = np.random.rand()
            if V < 0.5:
                L -= (R - L)
            else:
                R += (R - L)
            K -= 1
        return L, R

    def shrinkage(self, x, y, L, R):
        while True:
            u = np.random.rand()
            x1 = L + u * (R - L)
            if self.f(x1) > y:
                return x1
            else:
                if x1 < x:
                    L = x1
                else:
                    R = x1

    def test(self, x0, x1, w, y, L, R):
        L_hat = L
        R_hat = R
        D = False
        while R_hat - L_hat > 1.1 * w:
            M = (L_hat + R_hat) / 2
            if (x0 < M and x1 >= M) or (x0 >= M and x1 < M):
                D = True
            if x1 < M:
                R_hat = M
            else:
                L_hat = M
        if D and y >= self.f(L_hat) and y >= self.f(R_hat):
            return False
        else:
            return True

    def sample(self, w=1.0, m=10, p=5):
        """
        Perform one iteration of the slice sampling algorithm.

        Parameters:
        - w: Initial estimate of the slice width.
        - m: Limit on steps for the stepping-out procedure.
        - p: Limit on steps for the doubling procedure.

        Returns:
        - x_new: The new sample point.
        """
        x0 = self.x_0
        y = np.random.uniform(0, self.f(x0))
        # Choose an interval around x0
        L, R = self.stepping_out(x0, y, w, m)
        # Alternatively, you can use doubling:
        # L, R = self.doubling(x0, y, w, p)

        while True:
            # Propose a new point within the interval [L, R]
            x1 = self.shrinkage(x0, y, L, R)
            # Check if the new point is acceptable
            if self.test(x0, x1, w, y, L, R):
                # Accept the new point
                self.x_0 = x1
                return x1
            else:
                # Shrink the interval and repeat
                if x1 < x0:
                    L = x1
                else:
                    R = x1

    def sample_n(self, n, w=1.0, m=10, p=5):
        """
        Generate n samples using the slice sampling algorithm.

        Parameters:
        - n: Number of samples to generate.
        - w: Initial estimate of the slice width.
        - m: Limit on steps for the stepping-out procedure.
        - p: Limit on steps for the doubling procedure.

        Returns:
        - samples: A list containing n sample points.
        """
        samples = []
        for _ in range(n):
            x_new = self.sample(w, m, p)
            samples.append(x_new)
        return samples


# Define the target distribution (up to a proportionality constant)
def target_distribution(x):
    return np.exp(-x**2 / 2)

# Initialize the sampler
sampler = SliceSamplerUniDimensional(f=target_distribution, x_0=0.0)

# Generate 1000 samples
n_samples = 10000
samples = sampler.sample_n(n_samples)

# Plot the histogram of samples
plt.hist(samples, bins=30, density=True, alpha=0.7, label='Slice Sampling')

# Plot the true distribution for comparison
x = np.linspace(-4, 4, 1000)
plt.plot(x, (1/np.sqrt(2*np.pi)) * np.exp(-x**2 / 2), 'r-', label='True Distribution')
plt.legend()
plt.title('Slice Sampling of Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
