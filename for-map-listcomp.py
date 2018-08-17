import numpy as np
import pandas as pd
import timeit
from math import pow


N=100000
n_repeat=100
VALIDATE_ARRAYS = True

a = list(np.random.rand(N))
b = list(np.random.rand(N))

def add(a, b): return a+b
def dist1(a, b): return a*a + b*b
def dist2(a, b): return add(pow(a,2), pow(b,2))

functions = {'max': max,
             'sum': add,
             'dist1': dist1,
             'dist2': dist2,
             'power': pow,
             }

results = pd.DataFrame(index=['for+append', 'for+index', 
                              'map-lambda', 'map-nolambda', 'listcomp'], 
                       columns=list(functions.keys()), 
                       data=0.0)

for flabel, function in functions.items():

    statistics = np.empty((len(results.index), n_repeat))

    for r in range(n_repeat):

        t0 = timeit.default_timer()
        _cfi = a.copy()
        for i in range(N):
            _cfi[i] = function(a[i], b[i])
        t1 = timeit.default_timer()
        statistics[0, r] = t1-t0

        t0 = timeit.default_timer()
        _cfa = []
        for i in range(N):
            _cfa.append(function(a[i], b[i]))
        t1 = timeit.default_timer()
        statistics[1, r] = t1-t0

        t0 = timeit.default_timer()
        _cml = list(map(lambda i: function(a[i], b[i]), range(N)))
        t1 = timeit.default_timer()
        statistics[2, r] = t1-t0

        t0 = timeit.default_timer()
        _cmn = list(map(function, a, b))
        t1 = timeit.default_timer()
        statistics[3, r] = t1-t0

        t0 = timeit.default_timer()
        _clc = [function(a[i], b[i]) for i in range(N)]
        t1 = timeit.default_timer()
        statistics[4, r] = t1-t0

        if VALIDATE_ARRAYS and r==0:
            if not ((np.array(_cfa) == np.array(_cfi)).all() and
                    (np.array(_cfa) == np.array(_cml)).all() and 
                    (np.array(_cfa) == np.array(_cmn)).all() and 
                    (np.array(_cfa) == np.array(_clc)).all()):
                raise RuntimeError("Discrepancy in computation")

    for j, technique in enumerate(results.index):
        results.loc[technique, flabel] = np.min(statistics[j])


print(results)

