import cvxpy as cp
import time

# Define variables
x = cp.Variable()
y = cp.Variable()

### Objective
def U_bar(schedule, schedule_init, SINR, Q_inv, l, K):
    return -F_obj(schedule, SINR, K) + V_obj(schedule_init, SINR, Q_inv, l, K) + beta*(W(schedule) - E(schedule_init)) 

# Define objective function
objective = cp.Minimize(x**2 + 4*y**2)

# Define constraints
constraints = [x + y >= 3]

problem = cp.Problem(objective, constraints)
start = time.time()
N = 10000
# Solve the optimization problem
for i in range(N):
    problem.solve()
print('temps :',(time.time()-start)/N)

print("Using cvxpy:")
print("Optimal solution x:", x.value)
print("Optimal solution y:", y.value)
print("Optimal value:", problem.value)
