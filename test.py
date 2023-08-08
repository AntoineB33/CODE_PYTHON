import numpy as np

# Create a sample 3D array
array_3d = np.array([[[0,0,0],
                      [0,1, 6]],
                     
                     [[7, 8, 9],
                      [10, 11, 12]]])

# Sum along a specified axis (e.g., axis=2)
sum_result = np.sum(array_3d, axis=(1, 2))

print(sum_result)
