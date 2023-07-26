import numpy as np

# Example arrays
arr1 = np.array([0, 1, 0, 1, 0])
arr2 = np.array([1, 0, 1, 1, 0])

# Boolean indexing to select elements that satisfy the condition
selected = (arr1 == 0) &(arr2 == 1)

# Calculate the sum of the selected elements using np.sum
result = np.sum(selected)

print(result)