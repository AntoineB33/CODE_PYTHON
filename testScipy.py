import numpy as np
from scipy.optimize import minimize
import time

# Objective function
def objective_function(x):
    return x[0]**2 + 4*x[1]**2

# Constraint function
def constraint_function(x):
    return x[0] + x[1] - 3

# Initial guess
initial_guess = np.array([0.0, 0.0])

# Constraints definition
constraints = {'type': 'ineq', 'fun': constraint_function}

N = 10000
start = time.time()
# Solve the optimization problem
for i in range(N):
    result = minimize(objective_function, initial_guess, constraints=constraints)
print('temps :',(time.time()-start)/N)

print("Using scipy.optimize.minimize:")
print("Optimal solution:", result.x)
print("Optimal value:", result.fun)
