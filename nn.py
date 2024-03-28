import layers

class NN:
    def __init__(self, *l):
        self.layer_chain = l
        self.length = len(self.layer_chain)
        self.intermediate = {}

    def apply(self, vector):
        k = vector
        self.intermediate[k] = np.array()
        for i, layer in enumerate(self.layer_chain):
            k = 
