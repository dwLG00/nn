import numpy as np
from scipy.special import expit as sigmoid
from scipy.special import softmax

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

    def __repr__(self):
        replacement_str = ', '.join('%s' for _ in self.shape)
        return 'Layer(%s)' % (replacement_str % self.shape)

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

    def inverse(self, vector):
        pinv = np.linalg.pinv(self.array)
        return np.dot(pinv, vector)

    @classmethod
    def init_random(cls, shape):
        return cls(np.random.rand(*shape)*2 - 1)

    def __repr__(self):
        return 'MatrixLayer(%s, %s)' % self.shape

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

    def __repr__(self):
        return 'VectorLayer(%s)' % self.shape[0]

class Sigmoid(Layer):
    def __init__(self, n):
        self.shape = (n,)

    def apply(self, vector):
        return sigmoid(vector)

    def derivative(self, vector):
        return np.diag(dsigmoid(vector))

    def inverse(self, vector):
        return np.log(vector) - np.log(1 - vector)

    def __repr__(self):
        return 'Sigmoid(%s)' % self.shape

class ReLU(Layer):
    def __init__(self, n):
        self.shape = (n,)
        self.func = np.vectorize(relu)
        self.dfunc = np.vectorize(drelu)

    def apply(self, vector):
        return self.func(vector)

    def derivative(self, vector):
        return np.diag(self.dfunc(vector))

    def inverse(self, vector): #Not invertible, so we'll find a good "pseudoinverse"
        maxvalue = np.amax(vector)
        vec = np.copy(vector)
        vec[vec == 0] = -maxvalue
        return vec

    def __repr__(self):
        return 'ReLU(%s)' % self.shape

class Identity(Layer):
    def __init__(self, n):
        self.shape = (n,)

    def apply(self, vector):
        return vector

    def derivative(self, vector):
        return np.identity(self.shape[0])

    def inverse(self, vector):
        return vector

    def __repr__(self):
        return 'Identity(%s)' % self.shape

class Softmax(Layer):
    # thanks https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    def __init__(self, n):
        self.shape = (n,)

    def apply(self, vector):
        shift = vector - np.max(vector)
        return softmax(shift)

    def derivative(self, vector):
        sm_value = softmax(vector)
        deriv = np.outer(sm_value, sm_value)
        diagonalized = np.diag(sm_value)
        return diagonalized - deriv

    def inverse(self, vector):
        vec = np.copy(vector).astype('float')
        vec[vec == 0] = 10**(-10)
        ln = np.log(vec)
        lowest, highest = min(ln), max(ln)
        diffmid = (highest - lowest) / 2
        return ln + diffmid

    def __repr__(self):
        return 'Softmax(%s)' % self.shape

class DirectProduct(Layer):
    def __init__(self, *layers):
        self.shape = (
            sum(layer.shape[0] for layer in layers),
            sum(layer.shape[1] if len(layer.shape) > 1 else layer.shape[0] for layer in layers)
        )
        self.layers = layers

    def apply(self, vector):
        i = 0
        out = None
        for layer in self.layers:
            dimension = layer.shape[0]
            vec = vector[i:i + dimension]
            res = layer.apply(vec)
            if out == None:
                out = res
            else:
                out = np.concatenate([out, res])
            i += dimension
        return out

    def derivative(self, vector):
        i = 0
        out = None
        for layer in self.layers:
            dimension = layer.shape[0]
            vec = vector[i:i + dimension]
            deriv = layer.derivative(vec)
            if out == None:
                out = deriv
            else:
                out = np.concatenate([out, deriv])
            i += dimension
        return out

class Bifurcate(Layer):
    def __init__(self, input_size, n):
        self.shape = (input_size, n * input_size)
        self.n_copies = n
        identity = np.identity(input_size)
        identity_block = np.block([[identity] * n])
        zero_block = np.zeros((n * input_size, (n - 1) * input_size))
        self.matrix = np.block([identity_block, zero_block
