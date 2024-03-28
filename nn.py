import layers
import numpy as np

class NeuralNet:
    def __init__(self, *l):
        self.layers = l
        self.length = len(self.layers)
        self.intermediate = {}

    def clear(self):
        self.intermediate = {}

    def apply(self, vector):
        k = vector
        intermediate = []
        for layer in self.layers:
            k = layer.apply(k)
            intermediate.append(k)
        self.intermediate[str(vector.data)] = intermediate
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

            intermediates = self.intermediate[str(vector.data)][::-1]

            partial = None
            for j, layer in enumerate(self.layers[::-1]):
                if isinstance(layer, layers.WeightLayer):
                    if partial is None:
                        grad = layer.weight_derivative(vector)
                    else:
                        grad = np.tensordot(layer.weight_derivative(vector), partial, 0)
                    grads[j] += grad
                if partial is None: partial = layer.derivative(vector)
                else:
                    if i < len(intermediates):
                        partial = np.dot(layer.derivative(intermediates[j]), partial)
                    else:
                        partial = np.dot(layer.derivative(vector), partial)
            print('Grads %s: %s' % (i, grads))



        for i, layer in enumerate(self.layers[::-1]):
            if isinstance(layer, layers.WeightLayer):
                layer.apply_grad(2 * grads[i] / n_inputs)

        net_error /= n_inputs
        return grads, net_error
