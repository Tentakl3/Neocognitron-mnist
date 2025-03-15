import math

class VCCell:
    def __init__(self, d):
        #efficency of the unmodifiable excitatory synapses
        self.d = d

    def propagate(self, inputs):
        output = 0.0
        for k in range(inputs.shape[0]):
            for w in range(inputs[0].shape[0]):
                output += inputs[k][w] * self.d[w] 
        output = (1/(inputs.shape[0]))*output
        #calculation of the inhibitory cell
        return output