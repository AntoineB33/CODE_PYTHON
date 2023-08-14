import numpy as np

# Create a sample 3D matrix
matrix_3d = np.array([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])

# Sum along the second axis (axis=1) to get a 1D array
sum_array = np.sum(matrix_3d, axis=(0,2))

print(sum_array)