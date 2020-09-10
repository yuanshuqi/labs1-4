import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as ss
import numpy as np

distribs = {'normal': {'d': ss.norm, 'scale': 1},
            'cauchy': {'d': ss.cauchy, 'scale': 1},
            'laplace': {'d': ss.laplace, 'scale': 1 / (2 ** 0.5)},
            'poisson': {'d': ss.poisson, 'scale': 10},
            'uniform': {'d': ss.uniform, 'scale': 3 ** 0.5}}
distrib_names = ['normal', 'cauchy', 'laplace', 'poisson', 'uniform']


def distrib_sample(name, n):
    if name.lower() == 'normal':
        return distribs['normal']['d'].rvs(size=n)
    elif name.lower() == 'cauchy':
        return distribs['cauchy']['d'].rvs(size=n)
    elif name.lower() == 'laplace':
        return distribs['laplace']['d'].rvs(size=n, scale=distribs['laplace']['scale'])
    elif name.lower() == 'poisson':
        return distribs['poisson']['d'].rvs(size=n, mu=distribs['poisson']['scale'])
    elif name.lower() == 'uniform':
        return distribs['uniform']['d'].rvs(size=n, loc=-distribs['uniform']['scale'],
                                            scale=distribs['uniform']['scale'] * 2)
    else:
        return np.array([0])


def cdf(name, x):
    if name.lower() == 'normal':
        return distribs['normal']['d'].cdf(x)
    elif name.lower() == 'cauchy':
        return distribs['cauchy']['d'].cdf(x)
    elif name.lower() == 'laplace':
        return distribs['laplace']['d'].cdf(x, scale=distribs['laplace']['scale'])
    elif name.lower() == 'poisson':
        return distribs['poisson']['d'].cdf(x, mu=distribs['poisson']['scale'])
    elif name.lower() == 'uniform':
        return distribs['uniform']['d'].cdf(x, loc=-distribs['uniform']['scale'],
                                            scale=distribs['uniform']['scale'] * 2)
    else:
        return np.array([0])


def pdf(name, x):
    if name.lower() == 'normal':
        return distribs['normal']['d'].pdf(x)
    elif name.lower() == 'cauchy':
        return distribs['cauchy']['d'].pdf(x)
    elif name.lower() == 'laplace':
        return distribs['laplace']['d'].pdf(x, scale=distribs['laplace']['scale'])
    elif name.lower() == 'poisson':
        return distribs['poisson']['d'].pmf(x, mu=distribs['poisson']['scale'])
    elif name.lower() == 'uniform':
        return distribs['uniform']['d'].pdf(x, loc=-distribs['uniform']['scale'],
                                            scale=distribs['uniform']['scale'] * 2)
    else:
        return np.array([0])


def edf(name, size, intrvl):
    """Empirical distribution function"""
    smp = distrib_sample(name, size)
    smp.sort()
    meta_smp = {}
    unique_smp = np.unique(smp)
    meta_smp[unique_smp[0]] = np.sum(smp == unique_smp[0]) / size
    for i in range(1, len(unique_smp)):
        meta_smp[unique_smp[i]] = meta_smp[unique_smp[i - 1]] + np.sum(smp == unique_smp[i]) / size
    edf_y = []
    for i in range(len(intrvl)):
        less_than = np.ravel(np.extract(smp < intrvl[i], smp))
        edf_y.append(
            0 if len(less_than) == 0 else
            meta_smp[max(less_than)]
        )
    return edf_y


def pde(name, size, intrvl, k):
    """Probability density estimate"""
    smp = distrib_sample(name, size)
    df_sigma = pd.DataFrame(smp).std()
    sigma = df_sigma.to_numpy()[0]
    h = k * 1.06 * sigma * (size ** -0.2)

    pde_y = []
    for x in intrvl:
        sum_k = 0
        for xs in smp:
            sum_k += pdf('normal', (x - xs) / h)
        pde_y.append(sum_k / (size * h))
    return pde_y


ns = [20, 60, 100]
ks = [0.5, 1, 2]


def draw_df():
    for name in distrib_names:
        if name != 'poisson':
            a, b = -4, 4
        else:
            a, b = 6, 14
        x = np.arange(a, b, 0.05)
        df_axis = [a, b, -0.1, 1.1]
        for n in ns:
            fig, ax = plt.subplots(1, 1)
            plt.axis(df_axis)
            plt.plot(x, edf(name, n, x))
            plt.plot(x, cdf(name, x))
            ax.set_title(name.title() + ' n={}'.format(n))
            ax.set_ylabel('F(x)')
            ax.set_xlabel('x')
            ax.grid(axis='y')
            ax.legend(['edf', 'cdf'])
            plt.show()


def draw_pd():
    for name in ['uniform']:
        if name != 'poisson':
            a, b = -4, 4
        else:
            a, b = 6, 14
        x = np.arange(a, b, 0.01)
        df_axis = [a, b, -0.01, 1.01]
        for n in ns:
            fig, ax = plt.subplots(1, 1)
            plt.axis(df_axis)

            for k in ks:
                plt.plot(x, pde(name, n, x, k))
            if name == 'poisson':
                x = np.arange(a, b + 1, 1)
            plt.plot(x, pdf(name, x), color='black')

            ax.set_title(name.title() + ' n={}'.format(n))
            ax.set_ylabel('f(x)')
            ax.set_xlabel('x')
            ax.grid(axis='y')
            ax.legend(['k=0.5', 'k=1', 'k=2', 'f(x)'])
            plt.show()


# draw_df()
draw_pd()
