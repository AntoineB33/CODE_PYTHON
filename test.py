import numpy as np

ar = np.array([[1, 2, 3],[8,5,6]])
sub = np.array([1, 0])
ar[sub] = 2
print(ar)
