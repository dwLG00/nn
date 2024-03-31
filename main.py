from nn import NeuralNet
import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# A: 2 -> 5
# B: 5 -> 4
# C: 4 -> 1

A = layers.MatrixLayer.init_random((5, 2))
B = layers.MatrixLayer.init_random((4, 5))
C = layers.VectorLayer.init_random(4)

net = NeuralNet(A, layers.ReLU(5), B, layers.Sigmoid(4), C)

# Let z = x**2 + y
z = lambda x: x[0] + max(x[0], x[1]) - min(x[0], x[1] - 3)

def gen_data():
    data = np.random.rand(100, 2)
    expected = np.array([z(row) for row in data])
    return data, expected

#print('data: %s' % data)
#print('expected: %s' % expected)

'''
print('A: %s' % A.array)
print('B: %s' % B.array)
print('C: %s' % C.vector)

'''
errors = []
start = time.time()
for i in range(10000):
    data, expected = gen_data()
    grads, net_error = net.compute_grad(data, expected)
    print('Run %s' % i)
    #print('Weights: %s' % [layer.array for layer in [layer for layer in net.layers if isinstance(layer, layers.WeightLayer)]])
    #print('Grads: %s' % grads)
    print('Net error function: %s' % net_error)
    net.apply_grad(grads)
    net.clear()
    errors.append(net_error)

end = time.time()
print('Took %s' % (end - start))

plt.plot(errors)
plt.show()
