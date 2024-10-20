import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf

class SamplerVisualizer:
    @staticmethod
    def plot_distribution(samples, target_distribution, title=''):
        sns.histplot(samples, bins=50, kde=True, stat='density', label='Sampled Distribution', color='blue')
        x = np.linspace(min(samples), max(samples), 1000)
        y = target_distribution(x)
        y_normalized = y / np.trapz(y, x)
        plt.plot(x, y_normalized, 'r', label='Target Distribution', linewidth=2)
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('Density')
        plt.legend()
        plt.show()

    @staticmethod
    def plot_trace(samples, title=''):
        plt.figure(figsize=(12, 4))
        plt.plot(samples)
        plt.title(title)
        plt.xlabel('Iteration')
        plt.ylabel('Sample Value')
        plt.show()

    @staticmethod
    def plot_autocorrelation(samples, lags=50, title=''):
        plot_acf(samples, lags=lags)
        plt.title(title)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()
