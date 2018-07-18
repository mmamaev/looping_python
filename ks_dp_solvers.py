import numpy as np
import sys
import argparse


# This super straigthforward implementation turns out to consume 2GB of memory
# due to blunt initialization of a new grid line in line 14. Have an idea why?
# Nasdaq100 running time is about 3 minutes.

def solve_naive_inflated(capacity, items, weights, values):

    grid = [[0] * (capacity+1)]
    for item in range(items):
        grid.append([0] * (capacity+1))
        for k in range(capacity+1):
            if weights[item] > k:
                grid[item + 1][k] = grid[item][k]
            else:
                grid[item + 1][k] = max(grid[item][k], grid[item][k-weights[item]] + values[item])

    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# A slightly better straightforward solver. It consumes the expected 
# 400 MB of memory and saves some iterations in the inner for loop by starting
# at current item's weight instead of zero.
# Nasdaq100 running time is 180s.   

def solve_naive(capacity, items, weights, values):

    grid = [[0] * (capacity+1)]
    for item in range(items):
        grid.append(grid[item].copy())
        for k in range(weights[item], capacity+1):
            grid[item + 1][k] = max(grid[item][k], grid[item][k-weights[item]] + values[item])

    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# Same as above but the inner loop is broken down into inidividual operations
# to obtain visibility of their running times.
# Nasdaq100 running time is 317s.  

def solve_naive_detailed(capacity, items, weights, values):

    grid = [[0] * (capacity+1)]
    for item in range(items):
        grid.append(grid[item].copy())
        wi = weights[item]
        vi = values[item]
        for k in range(wi, capacity+1):
            a = grid[item][k]
            b = grid[item][k-wi] + vi
            if b > a:
                grid[item + 1][k] = b
            else:
                grid[item + 1][k] = a


    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# The "else" part of the inner loop above (lines 76, 77) is redundant
# but removing it does not make the function to run faster than `solve_naive`.
# Nasdaq100 running time is 254s.

def solve_naive_nomax(capacity, items, weights, values):

    grid = [[0] * (capacity+1)]
    for item in range(items):
        grid.append(grid[item].copy())
        wi = weights[item]
        vi = values[item]
        for k in range(wi, capacity+1):
            b = grid[item][k-wi] + vi
            if b > grid[item][k]:
                grid[item + 1][k] = b


    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# Using arrays from Python's standard library instead of lists does not help.
# Actually arrays make things slightly worse. 
# Nasdaq100 running time is 207s.

from array import array

def solve_naive_array(capacity, items, weights, values):

    weights = array('l', weights)
    values = array('l', values)
    grid = [array('l', [0] * (capacity+1))]
    for item in range(items):
        grid.append(array('l', grid[item]))
        for k in range(weights[item], capacity+1):
            grid[item + 1][k] = max(grid[item][k], grid[item][k-weights[item]] + values[item])

    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# In the above function the grid is a list of arrays.
# What if we flatten the grid and build it inside a long one-dimensional array?
# Alas, we get the worst running time ever. Too many multiplications in indices.
# Nasdaq100 running time is 337s.

def solve_naive_array_flat(capacity, items, weights, values):

    weights = array('l', weights)
    values = array('l', values)
    grid = array('l', (0 for _ in range((capacity+1)*(items+1))))
    for item in range(items):
        grid[(item+1)*(capacity+1):(item+2)*(capacity+1)] = grid[item*(capacity+1):(item+1)*(capacity+1)]
        for k in range(weights[item], capacity+1):
            grid[(item+1)*(capacity+1)+k] = max(grid[item*(capacity+1)+k], grid[item*(capacity+1)+k-weights[item]] + values[item])

    solution_value = grid[-1]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item*(capacity+1)+k] != grid[(item-1)*(capacity+1)+k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken    


# The internal "for" loop is substituted by "map".
# Nasdaq100 running time is 101s - almost two times faster than `solve_naive`.

def solve_map(capacity, items, weights, values):
    grid = [[0] * (capacity+1)]

    for item in range(items):
        grid.append(grid[item].copy())
        this_weight = weights[item]
        this_value = values[item]
        
        grid[item+1][this_weight:] = list(map(lambda k: max(grid[item][k], grid[item][k - this_weight] + this_value), range(this_weight, capacity+1)))
        

    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# I've heard that predefining a function for `map` instead of using 
# lambda can make things faster. No, it does not.
# Nasdaq100 running time is 102s.

def solve_map_nolambda(capacity, items, weights, values):

    def selector(k):
        nonlocal item, grid, this_weight, this_value
        return max(grid[item][k], grid[item][k - this_weight] + this_value)

    grid = [[0] * (capacity+1)]

    for item in range(items):
        grid.append(grid[item].copy())
        this_weight = weights[item]
        this_value = values[item]
        
        grid[item+1][this_weight:] = list(map(selector, range(this_weight, capacity+1)))
        

    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# The internal "for" loop is substituted by list comprehension.
# Nasdaq100 running time is 81s - noticeably faster than the map-based solver.

def solve_list_comp(capacity, items, weights, values):

    grid = [[0] * (capacity+1)]

    for item in range(items):
        grid.append(grid[item].copy())
        this_weight = weights[item]
        this_value = values[item]
        
        grid[item+1][this_weight:] = [max(grid[item][k], grid[item][k - this_weight] + this_value) for k in range(this_weight, capacity+1)]
        
    solution_value = grid[items][capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item-1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# This is a list comprehension-based solver but the grid is now a 2D numpy array.
# However, numpy arrays without numpy functions are a waste of time.
# Nasdaq100 running time is 123s. 

def solve_list_comp_numpy(capacity, items, weights, values):

    grid = np.empty((items + 1, capacity + 1), dtype=int)
    grid[0] = 0
    for item in range(items):
        grid[item+1] = grid[item]
        this_weight = weights[item]
        this_value = values[item]
        grid[item+1, this_weight:] = [max(grid[item, k], grid[item, k - this_weight] + this_value) for k in range(this_weight, capacity+1)]

    solution_value = grid[items, capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item, k] != grid[item-1, k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# This solver puts data into numpy arrays and uses numpy functions to process them.
# Nasdaq100 running time is 0.56s.

def solve_numpy_func(capacity, items, weights, values):

    grid = np.empty((items + 1, capacity + 1), dtype=int)
    grid[0] = 0

    for item in range(items):
        this_weight = weights[item]
        this_value = values[item]
        grid[item+1, :this_weight] = grid[item, :this_weight]
        temp = grid[item, :-this_weight] + this_value
        grid[item + 1, this_weight:] = np.where(temp > grid[item, this_weight:], temp, grid[item, this_weight:])

    solution_value = grid[items, capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item - 1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# An attempt to optimize the above by prealocating `temp` array.
# Surprisingly, it turns out slower: Nasdaq100 running time is 0.66s.

def solve_numpy_func_buffer(capacity, items, weights, values):

    grid = np.empty((items + 1, capacity + 1), dtype=int)
    grid[0] = 0
    temp = np.zeros(capacity+1, dtype=int)

    for item in range(items):
        this_weight = weights[item]
        this_value = values[item]
        grid[item+1, :this_weight] = grid[item, :this_weight]
        temp[:capacity+1-this_weight] = grid[item, :-this_weight] + this_value
        grid[item + 1, this_weight:] = np.where(temp[:-this_weight] > grid[item, this_weight:], temp[:-this_weight], grid[item, this_weight:])

    solution_value = grid[items, capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item - 1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


# Recursive solver - no `for` loops at all. Not scalable. 
# Just to prove it is neither faster. Nasdaq100 running time is 0.66s.

def solve_numpy_func_recur(capacity, items, weights, values):

    def calculate(item):
        nonlocal grid, weights, values
        this_weight = weights[item-1]
        this_value = values[item-1]
        if item == 0:
            grid[0] = np.zeros((capacity+1), dtype=int)
        else:
            calculate(item-1)
            grid[item, :this_weight] = grid[item-1, :this_weight]
            temp = grid[item-1, :-this_weight] + this_value
            grid[item, this_weight:] = np.where(temp > grid[item-1, this_weight:], temp, grid[item-1, this_weight:])


    grid = np.empty((items + 1, capacity + 1), dtype=int)
    calculate(items)

    solution_value = grid[items, capacity]
    solution_weight = 0
    taken = []
    k = capacity
    for item in range(items, 0, -1):
        if grid[item][k] != grid[item - 1][k]:
            taken.append(item - 1)
            k -= weights[item - 1]
            solution_weight += weights[item-1]

    return solution_value, solution_weight, taken


if __name__ == '__main__':

    solvers = {
        'naive-i': solve_naive_inflated,
        'naive': solve_naive,
        'naive-detailed': solve_naive_detailed,
        'naive-nomax': solve_naive_nomax,
        'naive-array': solve_naive_array,
        'naive-array-flat': solve_naive_array_flat,
        'map': solve_map,
        'map-nolambda': solve_map_nolambda,
        'listcomp': solve_list_comp,
        'listcomp-numpy': solve_list_comp_numpy,
        'numpy': solve_numpy_func,
        'numpy-b': solve_numpy_func_buffer,
        'numpy-r': solve_numpy_func_recur,

    }

    parser = argparse.ArgumentParser(
        description='Implementations of dynamic programming solutions '
                    'of knapsack problem')
    parser.add_argument('-v', action='store_true', default=False,
                        help='Verbose output. Otherwise only solution value is printed')
    parser.add_argument('-f', default='nasdaq100list.csv', metavar='filename',
                        help='Name of CSV data file (default: %(default)s). '
                             'Data format: label, weight, value. No header line.')
    parser.add_argument('solver', choices=list(solvers.keys()),
                        help='Solver implementation. Choose from %(choices)s')
    parser.add_argument('-m', action='store_true', default=False,
                        help='Engage line memory profiler')
    parser.add_argument('-t', action='store_true', default=False,
                        help='To engage line time profiler run '
                        '"kernprof -v -l %(prog)s -t <solver> <other_options>"')
    args = parser.parse_args()
    verbose = args.v
    datafilename = args.f
    solver = solvers[args.solver]
    profile_mem = args.m
    profile_time = args.t


    labels = []
    weights = []
    values = []
    items = 0
    capacity = 1000000

    datafilename = 'nasdaq100list.csv'
    with open(datafilename, 'r') as file:
        data = file.read()
    for line in data.split('\n'):
        symbol, price, target = line.split(',')
        labels.append(symbol)
        weights.append(int(float(price)*100))
        values.append(int(float(target)*100))
        items +=1

    if verbose:
        print ("Got data: {} items".format(items))

    if profile_mem:
        from memory_profiler import profile

    if profile_mem or profile_time:
        solver = profile(solver)

    solution_value, solution_weight, taken = \
        solver(capacity, items, weights, values)

    
    if verbose:
        print("Solution_value: {}".format(solution_value))
        print("Solution weight: {}\nTook {} items\nItems taken: {}\n".
              format(solution_weight, len(taken),
                     ", ".join([labels[i] for i in sorted(taken)])))
    else:
        print(solution_value)
