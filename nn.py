import layers
import numpy as np
import math

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

    def compute_grad(self, inputs, exp_out, debug=False):
        # initial pass-through
        n_inputs = inputs.shape[0]
        grads = [0 for _ in range(self.length)]

        net_error = 0

        for i, vector in enumerate(inputs):
            exp = exp_out[i]
            res = self.apply(vector)
            error = res - exp
            #print('res, exp: %s, %s' % (res, exp))
            net_error += error**2

            intermediates = self.intermediate[str(vector.data)][::-1]
            intermediates.append(vector)

            partial = None
            for j, layer in enumerate(self.layers[::-1]):
                #if partial is None: print('Partial shape: None')
                #else: print('Partial shape: %s' % partial.shape)
                if isinstance(layer, layers.WeightLayer):
                    if partial is None:
                        grad = error * layer.weight_derivative(intermediates[j+1])
                    else:
                        #weight_derivative = layer.weight_derivative(intermediates[j+1])
                        #print('Weight derivative shape: %s' % (weight_derivative.shape,))
                        #grad = np.tensordot(weight_derivative, partial, 1)
                        grad = error * np.tensordot(intermediates[j+1], partial, 0)
                    #print('Grad shape: %s' % (grad.shape,))
                    grads[j] += grad
                if partial is None: partial = layer.derivative(vector)
                else:
                    if j < len(intermediates):
                        partial = np.dot(layer.derivative(intermediates[j+1]), partial)
                    else:
                        partial = np.dot(layer.derivative(vector), partial)
            #print('Grads %s: %s' % (i, grads))
            #print('Passthrough %s' % i)
            #print('Intermediate: %s' % self.intermediate[str(vector.data)])
            #print('Grad shape: %s' % ([0 if g is 0 else g.shape for g in grads]))

        net_error /= n_inputs
        #rms = math.sqrt(net_error)
        #return [grad / n_inputs for grad in grads], rms
        return [grad / n_inputs for grad in grads], net_error

    def compute_grad_multi(self, inputs, exp_out, debug=False):
        n_inputs = inputs.shape[0]
        grads = [0 for _ in range(self.length)]

        net_error = 0

        for i, vector in enumerate(inputs):
            if debug: print('Datapoint %s' % i)
            exp = exp_out[i]
            res = self.apply(vector)
            error_vec = res - exp
            error = np.dot(error_vec, error_vec)
            net_error += error

            intermediates = self.intermediate[str(vector.data)][::-1]
            intermediates.append(vector)

            partial = error_vec
            for j, layer in enumerate(self.layers[::-1]):
                if debug: print('Layer %s: %s, partial: %s' % (j, layer, partial))
                if isinstance(layer, layers.WeightLayer):
                    grad = np.tensordot(intermediates[j+1], partial, 0)
                    grads[j] += grad
                if j < len(intermediates):
                    derivative = layer.derivative(intermediates[j+1])
                    if debug: print('Computed derivative on %s' % intermediates[j+1])
                else:
                    derivative = layer.derivative(vector)
                    if debug: print('Computed derivative on %s' % vector)
                if debug: print('Derivative: %s' % derivative)
                partial = np.dot(derivative, partial)
                if debug: input()

        net_error /= n_inputs
        #rms = math.sqrt(net_error)
        #return [grad / n_inputs for grad in grads], rms
        return [grad / n_inputs for grad in grads], net_error

    def apply_grad(self, grads, stepsize=0.01):
        for i, layer in enumerate(self.layers[::-1]):
            if isinstance(layer, layers.WeightLayer):
                layer.apply_grad(grads[i] * stepsize)
