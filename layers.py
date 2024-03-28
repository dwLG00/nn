import numpy as np
from scipy.special import expit as sigmoid

def dsigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

class Layer:
    def __init__(self, shape):
        self.shape = shape

    def apply(vector):
        '''Takes in a vector of size shape[1] and returns a vector of size shape[0]'''
        return

    def derivative(self):
        '''Derivative wrt input'''
        return

class WeightLayer(Layer):
    def weight_derivative(self):
        pass

    @classmethod
    def init_random(cls, size):
        pass

class MatrixLayer(WeightLayer):
    def __init__(self, nm_array):
        # (n, m) array
        self.shape = nm_array.shape
        self.array = nm_array

    def apply(vector):
        return np.dot(self.array, vector)

    def derivative(self):
        return self.array

    def weight_derivative(vector):
        return np.tensordot(np.identity(shape[1]), vector, 0)

    @classmethod
    def init_random(cls, shape):
        cls(np.empty(shape))

class VectorLayer(WeightLayer):
    def __init__(self, vector):
        self.shape = vector.shape
        self.vector = vector

    def apply(vector):
        return np.vdot(self.vector, vector)

    def derivative(vector):
        return self.vector

    def weight_derivative(vector):
        return vector

    @classmethod
    def init_random(cls, size):
        cls(np.empty((size,)))

class Sigmoid(Layer):
    def __init__(self, n):
        self.shape = (n,)

    def apply(vector):
        return sigmoid(vector)

    def derivative(vector):
        return np.diag(dsigmoid(vector))
