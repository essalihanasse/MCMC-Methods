# MCMC Samplers 🧮🔥

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](#-requirements)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Contributors](https://img.shields.io/github/contributors/yourusername/mcmc_samplers)](https://github.com/yourusername/mcmc_samplers/graphs/contributors)
[![Forks](https://img.shields.io/github/forks/yourusername/mcmc_samplers.svg?style=social)](https://github.com/yourusername/mcmc_samplers/network/members)
[![Stars](https://img.shields.io/github/stars/yourusername/mcmc_samplers.svg?style=social)](https://github.com/yourusername/mcmc_samplers/stargazers)

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
- [Requirements](#-requirements)
- [Usage](#-usage)
  - [Quick Start](#-quick-start)
  - [Examples](#-examples)
    - [Sampling from a Univariate Distribution](#sampling-from-a-univariate-distribution)
    - [Adaptive Metropolis-Hastings Sampling](#adaptive-metropolis-hastings-sampling)
    - [Sampling from a Multivariate Distribution](#sampling-from-a-multivariate-distribution)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)
- [Acknowledgments](#-acknowledgments)
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
