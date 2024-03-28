from nn import NeuralNet
import layers
import numpy as np

# A: 5 -> 10
# B: 10 -> 7
# C: 7 -> 1

A = layers.MatrixLayer.init_random((10, 5))
B = layers.MatrixLayer.init_random((7, 10))
C = layers.VectorLayer.init_random(7)

net = NeuralNet(A, layers.Sigmoid(10), B, layers.Sigmoid(7), C)

# Let z = Mx * y be the function we want to approximate
M = np.array([[1, 4, 6, -2, 3], [5, -2, 9, -5, 4], [3, 4, -7, 2, 3], [-4, 2, -6, 8, 3], [1, -8, 4, 5, -2]])
y = np.array([4, 1, -5, 2, 3])
z = lambda x: np.vdot(y, np.dot(M, x))

data = np.identity(5)
expected = np.array([z(row) for row in data])

print('data: %s' % data)
print('expected: %s' % expected)

'''
print('A: %s' % A.array)
print('B: %s' % B.array)
print('C: %s' % C.vector)

'''
grads, net_error = net.compute_grad(data, expected)
print('Net error function: %s' % net_error)
print('Gradients: %s' % grads)
