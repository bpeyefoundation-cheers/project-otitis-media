import numpy as np

array = np.array([1, 2, 3, 4, 5])
print(array)

arr1 = np.array([1, 4, 5, 7])
arr2 = np.array([2, 5, 7, 8])
arr3 = np.add(arr1, arr2)
print(arr3)
arr4 = np.dot(arr1, arr2)
print(arr4)
arr5 = np.array([[1, 2, 3], [2, 3, 4]])
print(arr5)
print(np.shape(arr5))
print(np.size(arr5))
print(np.ndim(arr5))
arr6 = np.array([[1, 0, 3], [1, 3, 4], [4, 2, 0]], dtype=np.int32)
print(arr6)
# print(np.multiply(arr5,arr6))
print(np.matmul(arr5, arr6))
print(isinstance(arr5, np.ndarray))
print(arr5.itemsize)
print(np.random.random())
print(np.random.randn(2, 5))
print(np.vstack((arr5, arr6)))
print(np.arange(10).reshape(5, 2))
