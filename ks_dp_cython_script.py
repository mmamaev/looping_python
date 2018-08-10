import pyximport; pyximport.install()
import ks_dp_cython
import argparse
import numpy as np
import timeit


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Dynamic programming solution '
                    'of knapsack problem implemented in Cython')
    parser.add_argument('-v', action='store_true', default=False,
                        help='Verbose output. Otherwise only solution value is printed')
    parser.add_argument('-f', default='nasdaq100list.csv', metavar='filename',
                        help='Name of CSV data file (default: %(default)s). '
                             'Data format: label, weight, value. No header line.')
    parser.add_argument('-c', default=1000000, metavar='capacity',
                        help='Knapsack capacity')
    parser.add_argument('-n', default=10, metavar='number',
                        help='N of repetitions (to measure running time)')
    parser.add_argument('-o', action='store_true', default=False,
                        help='Optimize array indexing')
    parser.add_argument('-t', action='store_true', default=False,
                        help='Print running time (sec)')
    args = parser.parse_args()
    verbose = args.v
    datafilename = args.f
    capacity = int(args.c)
    n = int(args.n)
    optimize_indexing = args.o
    print_time = args.t

    solver = ks_dp_cython.solve_cython
    if optimize_indexing:
        solver = ks_dp_cython.solve_cython_optim

    labels = []
    weights = []
    values = []
    items = 0

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

    timers = np.empty((n), dtype=float) 
    for i in range(n):
        timer0 = timeit.default_timer()
        solution_value, solution_weight, taken = \
            solver(capacity, items, np.array(weights), np.array(values))
        timer1 = timeit.default_timer()
        timers[i] = timer1 - timer0

    if verbose:
        print("Solution_value: {}".format(solution_value))
        print("Solution weight: {}\nTook {} items\nItems taken: {}\n".
              format(solution_weight, len(taken),
                     ", ".join([labels[i] for i in sorted(taken)])))
    else:
        print(solution_value)

    if print_time:
        print("Running time {:.5f} sec.".format(np.mean(timers)))

