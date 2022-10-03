import numpy as np

# np.array([[], [], []])

# C = np.array([[1,2,1,2], [4,1,-1,-4]])
# C = np.array([[1,2], [2,4]])
# B = np.array([[0,3], [1,-1], [2,1], [5,2]])
# C = A.dot(B)

A = np.array([[1,1,-1,-1], [2,5,-7,-5], [2,-1,1,3], [5,2,-4,-2]])
B = np.linalg.inv(A)

b = np.array([[1],[-2],[4],[6]])
# b = np.array([1,-2,4,6])
# b = np.transpose(np.array([1,-2,4,6]))

# D = np.linalg.inv(C)



print(B)
print(b)
print(B.dot(b))
# print(np.transpose(b))
# print(D)

# print(np.linalg.det(A))
# print(np.linalg.det(C))

# C = B.dot()