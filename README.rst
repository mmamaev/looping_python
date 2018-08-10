************
Python loop optimization for numerical calculations
************

This is the companion code for article `"If you have slow loops in Python, you can fix it…until you can’t" <https://medium.freecodecamp.org/if-you-have-slow-loops-in-python-you-can-fix-it-until-you-cant-3a39e03b6f35>`_

- ks_dp_solvers.py - a number of solver functions implementing dynamic programming algorithm to solve the knapsack problem. Solvers employ different approaches to looping through the array of data. 
- ks_dp_solvers_profiles.txt - outputs of line profiler for the above implementations
- ks_dp_cython.pyx - a straightforward solver (two nested for loops) based on cython
- ks_dp_cython_script.py - the script to run ks_dp_cython.pyx
- ks_dp_naive_solver.go - a straightforward solver (two nested for loops) coded in Golang
- nasdaq100list.csv - data file (Nasdaq 100 list of stock prices and price estimates)
- ks_dp_example.pdf - an annotated illustration of the dynamic programming algorithm used to solve the knapsack problem
