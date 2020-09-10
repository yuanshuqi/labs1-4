import numpy as np
import scipy.stats as ss
from tabulate import tabulate
import math


def values(rv, n):
    rv.sort()

    def z_p(p):
        return rv[int(n * p) - 1] if (n * p) % 1 == 0 else rv[math.floor(n * p)]

    r = n // 4

    return np.array([sum(rv) / n, (rv[n // 2 - 1] + rv[n // 2]) / 2 if n % 2 == 0 else rv[(n - 1) // 2],
                     (rv[0] + rv[n - 1]) / 2, (z_p(1 / 4) + z_p(3 / 4)) / 2, sum(rv[r:n - r - 1]) / (n - 2 * r)])


sizes = [10, 100, 1000]
reps = 1000
columns = ['n', 'mean', 'median', 'z_r', 'z_q', 'z_tr']
names = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']

for size in sizes:
    sample = np.array([[0, 0, 0, 0, 0]])
    for _ in range(reps):
        sample = np.append(sample, [values(ss.norm.rvs(size=size), size)], axis=0)
        sample = np.append(sample, [values(ss.cauchy.rvs(size=size), size)], axis=0)
        sample = np.append(sample, [values(ss.laplace.rvs(scale=(1 / (2 ** 0.5)), size=size), size)], axis=0)
        sample = np.append(sample, [values(ss.poisson.rvs(mu=10, size=size), size)], axis=0)
        sample = np.append(sample, [values(ss.uniform.rvs(loc=-(3 ** .5), scale=2 * (3 ** .5), size=size), size)], axis=0)
    es = ['E']
    ds = ['D']
    for j in range(len(columns) - 1):
        met = np.take(sample, j, axis=1)
        e = sum(met) / reps
        es.append(e)

        met = [x ** 2 for x in met]
        d = (sum(met) / reps) - e ** 2
        ds.append(d)
    res = np.array([es, ds])
    print("n={}".format(size))
    print(tabulate(res, headers=columns, floatfmt=".6f", tablefmt='github'))

