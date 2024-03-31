import numpy as np
from scipy.special import expit as sigmoid

def dsigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    if x > 0: return x
    return 0

def drelu(x):
    if x > 0: return 1
    return 0

class Layer:
    def __init__(self, shape):
        self.shape = shape

    def apply(self, vector):
        '''Takes in a vector of size shape[1] and returns a vector of size shape[0]'''
        return

    def derivative(self):
        '''Derivative wrt input'''
        return

class WeightLayer(Layer):
    def weight_derivative(self, vector):
        pass

    def apply_grad(self, ndarray):
        pass

    @classmethod
    def init_random(cls, size):
        pass

class MatrixLayer(WeightLayer):
    def __init__(self, nm_array):
        # (n, m) array
        self.shape = nm_array.shape
        self.array = nm_array

    def apply(self, vector):
        return np.dot(self.array, vector)

    def derivative(self, vector):
        return np.transpose(self.array)

    def weight_derivative(self, vector):
        return np.tensordot(np.identity(self.shape[0]), vector, 0)

    def apply_grad(self, matrix):
        self.array -= np.transpose(matrix)

    @classmethod
    def init_random(cls, shape):
        return cls(np.random.rand(*shape))

class VectorLayer(WeightLayer):
    def __init__(self, vector):
        self.shape = vector.shape
        self.array = vector

    def apply(self, vector):
        return np.vdot(self.array, vector)

    def derivative(self, vector):
        return self.array

    def weight_derivative(self, vector):
        return vector

    def apply_grad(self, vector):
        self.array -= vector

    @classmethod
    def init_random(cls, size):
        return cls(np.random.rand(size))

class Sigmoid(Layer):
    def __init__(self, n):
        self.shape = (n,)

    def apply(self, vector):
        return sigmoid(vector)

    def derivative(self, vector):
        return np.diag(dsigmoid(vector))

class ReLU(Layer):
    def __init__(self, n):
        self.shape = (n,)
        self.func = np.vectorize(relu)
        self.dfunc = np.vectorize(drelu)

    def apply(self, vector):
        return self.func(vector)

    def derivative(self, vector):
        return self.dfunc(vector)

