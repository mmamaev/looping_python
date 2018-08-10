import numpy as np
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t


def solve_cython(long capacity, 
                 long items, 
                 np.ndarray[DTYPE_t] weights, 
                 np.ndarray[DTYPE_t] values):

    cdef np.ndarray[DTYPE_t, ndim=2] grid = np.empty((items + 1, capacity + 1), dtype=DTYPE)
    grid[0] = 0

    cdef long this_weight
    cdef long this_value
    cdef long item
    cdef long k

    for item in range(items):
        this_weight = weights[item]
        this_value = values[item]
        grid[item + 1, :this_weight] = grid[item, :this_weight]

        for k in range(this_weight, capacity + 1):
            grid[item + 1, k] = max(grid[item, k], 
                                    grid[item, k - this_weight] + this_value)

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

# Below is exactly the same function under differnet name, compiled with options
# Could not figure how to DRY but have it simple and fast

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def solve_cython_optim(long capacity, long items, 
                       np.ndarray[DTYPE_t] weights, 
                       np.ndarray[DTYPE_t] values):


    cdef np.ndarray[DTYPE_t, ndim=2] grid = np.empty((items + 1, capacity + 1), dtype=DTYPE)
    grid[0] = 0

    cdef long this_weight
    cdef long this_value
    cdef long item
    cdef long k

    for item in range(items):
        this_weight = weights[item]
        this_value = values[item]
        grid[item + 1, :this_weight] = grid[item, :this_weight]

        for k in range(this_weight, capacity + 1):
            grid[item + 1, k] = max(grid[item, k], 
                                    grid[item, k - this_weight] + this_value)

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

