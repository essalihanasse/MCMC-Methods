# MCMC Samplers 🧮🔥

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](#-requirements)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

A comprehensive implementation of Monte Carlo Markov Chain (MCMC) algorithms, focusing on the **Metropolis-Hastings** algorithm and its advanced variants, all neatly organized into reusable and modular classes. Perfect for enthusiasts, students, and professionals looking to delve deep into the world of MCMC methods.

---

## 🌟 Features

- **Base Metropolis-Hastings Sampler**: Standard implementation with symmetric proposal distributions.
- **Kernel-Based Metropolis-Hastings Sampler**: Utilize asymmetric proposal distributions using kernel functions like the Laplace distribution.
- **Adaptive Metropolis-Hastings Sampler**: Dynamically adapts the proposal distribution during sampling using Kernel Density Estimation (KDE).
- **Multivariate Sampler**: Extend sampling capabilities to multivariate target distributions.
- **Visualization Utilities**: Handy functions for plotting distributions, trace plots, and autocorrelation to analyze sampler performance.

---

## 📖 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Additional Notes](#-additional-notes)
- [Enjoy!](#-enjoy)

---

## 🚀 Introduction

This repository offers a deep dive into Monte Carlo Markov Chain methods, with a special focus on the Metropolis-Hastings algorithm. Whether you're a researcher, student, or data science enthusiast, this project provides a solid foundation and practical tools to explore and implement MCMC algorithms in your own work.

---

## 🔧 Installation

Clone the repository and install the required packages using `pip`:

```bash
git clone https://github.com/essalihanasse/MCMC-Methods.git
cd MCMC-Methods
pip install -r requirements.txt
```
## 📝 Usage
⚡ Quick Start
```python
from mcmc_samplers import MetropolisHastingsSampler, SamplerVisualizer
import numpy as np

# Define the target distribution
def target_distribution(x):
    return 0.3 * np.exp(-0.5 * ((x - 2)/0.5)**2) + \
           0.7 * np.exp(-0.5 * ((x + 2)/0.8)**2)

# Define the proposal distribution
def gaussian_proposal(x, scale):
    return np.random.normal(loc=x, scale=scale)

# Set parameters
initial_state = 0.0
num_samples = 5000
burn_in = 1000
proposal_params = {'scale': 1.0}

# Instantiate and run the sampler
sampler = MetropolisHastingsSampler(
    target_distribution=target_distribution,
    proposal_distribution=gaussian_proposal,
    proposal_params=proposal_params
)

samples = sampler.sample(
    initial_state=initial_state,
    num_samples=num_samples,
    burn_in=burn_in
)

print(f'Acceptance Rate: {sampler.acceptance_rate:.2%}')

# Visualize the results
SamplerVisualizer.plot_distribution(
    samples=sampler.samples,
    target_distribution=target_distribution,
    title='Metropolis-Hastings Sampling'
)

```
## 📄 License
This project is licensed under the terms of the MIT License. See the LICENSE file for details.
