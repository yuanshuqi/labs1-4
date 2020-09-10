import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

ns = [10, 50, 1000]


def main():
    for n in ns:
        fig, ax = plt.subplots()
        sample = ss.norm.rvs(size=n)

        data = pd.DataFrame(sample)
        data.plot.hist(ax=ax, density=True, legend=False, title='Normal Numbers n={}'.format(n), color='#CCA365',
                       edgecolor='black', linewidth=1)
        x = np.linspace(min(sample), max(sample), 200)
        dens = pd.DataFrame(ss.norm.pdf(x), x)
        dens.plot.line(ax=ax, legend=False, color='#110AE1', linewidth=2)

        ax.set_ylabel('Density')
        ax.set_xlabel('Normal Numbers')
        ax.grid(axis='y')
        plt.show()

    for n in ns:
        fig, ax = plt.subplots()
        sample = ss.cauchy.rvs(size=n)

        data = pd.DataFrame(sample)
        data.plot.hist(ax=ax, density=True, legend=False, title='Cauchy Numbers n={}'.format(n), color='#CCA365',
                       edgecolor='black', linewidth=1)
        x = np.linspace(min(sample), max(sample), 200)
        dens = pd.DataFrame(ss.cauchy.pdf(x), x)
        dens.plot.line(ax=ax, legend=False, color='#110AE1', linewidth=2)

        ax.set_ylabel('Density')
        ax.set_xlabel('Cauchy Numbers')
        ax.grid(axis='y')
        plt.show()

    for n in ns:
        fig, ax = plt.subplots()
        sample = ss.laplace.rvs(size=n, scale=1 / (2 ** 0.5))

        data = pd.DataFrame(sample)
        data.plot.hist(ax=ax, density=True, legend=False, title='Laplace Numbers n={}'.format(n), color='#CCA365',
                       edgecolor='black', linewidth=1)
        x = np.linspace(min(sample), max(sample), 200)
        dens = pd.DataFrame(ss.laplace.pdf(x, scale=1 / (2 ** 0.5)), x)
        dens.plot.line(ax=ax, legend=False, color='#110AE1', linewidth=2)

        ax.set_ylabel('Density')
        ax.set_xlabel('Laplace Numbers')
        ax.grid(axis='y')
        plt.show()

    for n in ns:
        fig, ax = plt.subplots()
        sample = ss.poisson.rvs(size=n, mu=10)

        data = pd.DataFrame(sample)
        data.plot.hist(ax=ax, density=True, legend=False, title='Poisson Numbers n={}'.format(n), color='#CCA365',
                       edgecolor='black', linewidth=1)
        x = np.arange(min(sample) - 3, max(sample) + 3, 1)
        dens = pd.DataFrame(ss.poisson.pmf(x, mu=10), x)
        dens.plot.line(ax=ax, legend=False, color='#110AE1', linewidth=2)

        ax.set_ylabel('Density')
        ax.set_xlabel('Poisson Numbers')
        ax.grid(axis='y')
        plt.show()

    for n in ns:
        fig, ax = plt.subplots()
        scale = 3 ** 0.5
        sample = ss.uniform.rvs(size=n, loc=-scale, scale=scale * 2)

        data = pd.DataFrame(sample)
        data.plot.hist(ax=ax, density=True, legend=False, title='Uniform Numbers n={}'.format(n), color='#CCA365',
                       edgecolor='black', linewidth=1)
        x = np.linspace(min(sample), max(sample), 200)
        dens = pd.DataFrame(ss.uniform.pdf(x, loc=-scale, scale=scale * 2), x)
        dens.plot.line(ax=ax, legend=False, color='#110AE1', linewidth=2)

        ax.set_ylabel('Density')
        ax.set_xlabel('Uniform Numbers')
        ax.grid(axis='y')
        plt.show()


main()
