import layers
import numpy as np

class NeuralNet:
    def __init__(self, *l):
        self.layer_chain = l
        self.length = len(self.layer_chain)
        self.intermediate = {}

    def clear(self):
        self.intermediate = {}

    def apply(self, vector):
        k = vector
        intermediate = np.empty((self.length,))
        for i, layer in enumerate(self.layer_chain):
            k = layer.apply(k)
            intermediate[i] = k
        self.intermediate[vector] = intermediate
        return k

    def compute_grad(self, inputs, exp_out):
        # initial pass-through
        n_inputs = inputs.shape[0]
        grads = [0 for _ in range(n_inputs)]

        net_error = 0

        for i, vector in enumerate(inputs):
            exp = exp_out[i]
            res = self.apply(vector)
            error = res - exp
            net_error += error**2

            partial = None
            for i, layer in enumerate(self.layers[::-1]):
                if isinstance(layer, WeightLayer):
                    if not partial:
                        grad = layer.weight_derivative(vector)
                    else:
                        grad = np.tensordot(layer.weight_derivative(vector), partial, 0)
                    grads[i] += grad
                if not partial: partial = layer.derivative(vector)
                else:
                    partial = np.dot(layer.derivative(vector), partial)

        for i, layer in enumerate(self.layers[::-1]):
            if isinstance(layer, WeightLayer):
                layer.apply_grad(2 * grads[i] / n_inputs)

        net_error /= n_inputs
        return grads, net_error
