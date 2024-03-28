from nn import NeuralNet
import layers
import numpy as np

# A: 2 -> 2
# B: 2 -> 2
# C: 2 -> 1

A = layers.MatrixLayer.init_random((2, 2))
B = layers.MatrixLayer.init_random((2, 2))
C = layers.VectorLayer.init_random(2)

net = NeuralNet(A, layers.Sigmoid(2), B, layers.Sigmoid(2), C)

# Let z = x + y
z = lambda x: x[0] + x[1]

data = np.random.rand(100, 2)
expected = np.array([z(row) for row in data])

print('data: %s' % data)
print('expected: %s' % expected)

'''
print('A: %s' % A.array)
print('B: %s' % B.array)
print('C: %s' % C.vector)

'''
#for i in range(50):
i = 0
while True:
    i += 1
    grads, net_error = net.compute_grad(data, expected)
    print('Run %s' % i)
    print('Weights: %s', [])
    print('Grads: %s' % grads)
    print('Net error function: %s' % net_error)
    net.apply_grad(grads)
    net.clear()
    if input():
        break
#print('Gradients: %s' % grads)
