# neural net

Small neural net framework, written using numpy. Intended as a learning exercise.

# Example Usage
```py
from nn import nn, layers
import numpy as np

# Data
f = lambda x: round(sum(x)) #this will be our classifier heuristic
training_input = np.random.rand(10000, 3)
training_labels = np.zeros((10000, 4))
for i, x in enumerate(training_input):
    training_labels[i][f(x)] = 1

test_input = np.random.rand(1000, 3)
test_labels = np.zeros((1000, 4))
for i, x in enumerate(test_input):
    test_labels[i][f(x)] = 1

# classifier
net = nn.NeuralNet(
    layers.MatrixLayer.init_random((10, 3)),
    layers.ReLU(10),
    layers.MatrixLayer.init_random((4, 10)),
    layers.Softmax(4)
)

net.autotrain(
    np.split(training_input, 100),
    np.split(training_labels, 100),
    multi=True,
    stopfunction=nn.limit(50)
)

successes = 0
for x, y in zip(test_input, test_labels):
    fx = net.apply(x)
    if y[np.argmax(fx)] == 1: successes += 1

print('%s successes out of 1000 (%s%%)' % (successes, successes / 10))
```
