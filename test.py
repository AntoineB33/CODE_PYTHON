import numpy as np

ar = np.array([1, 2, 3, 4, 5, 6, 7])
sub = np.array([0, 1, 0, 1, 0, 0, 1])
ar[sub] = 2
print(ar)
