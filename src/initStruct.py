import pickle as pk
import random
import math
import matplotlib.pyplot as plt
import numpy as np

class InitStruct:
    def __init__(self):
        #These are the original parameters of Neocognitron from the 1980 paper
##########################################################################
#                        DON'T CHANGE THESE                              #
##########################################################################
        self.NUM_LAYERS = 3

        #numbers input size
        self.INPUT_LAYER_SIZE = 16
##########################################################################
#                              VARIABLE                                  #
##########################################################################
        self.S_LAYER_SIZES = [16, 8, 4]
        self.C_LAYER_SIZES = [10, 6, 1]
        
        self.PLANES_PER_LAYER = [24, 24, 24]

        self.S_WINDOW_SIZE = [5, 5, 2]
        self.C_WINDOW_SIZE = [5, 5, 2]

        self.S_COLUMN_SIZE = [5, 5, 2]

        # Q -> speed of reinforcement
        self.Q = [1.0, 16.0, 16.0]

        # R -> efficiency of inhibitory signals
        self.R = [4.0, 1.5, 1.5]

        self.gamma = [.11, .42, .06]
        self.delta = [.49, .60, .90]
        self.delta_bar = [.39, .68, .90]

        self.A_RANGE = [0.55, 0.45, 0.35]

        self.generateC()
        self.generateD()

        # C -> strength of the fixed excitatory connections for V cells 
        # monotonically decreasing in size of receptive field 
        # self.C = [.6, .2, .06, .04]
        # self.C = [.6, .24, .06]

        # D -> strength of the fixed excitatory connections for C cells 
        # monotonically decreasing in size of receptive field 
        # self.D = [.6, .2, .06, .04]
        # self.D = [.6, .24, .06]
    
    """Unmodifiable excitatory synapses monotonically deacreasing function"""
    """Is the same for the C values and the D values"""
    def generateC(self):
        self.C = []
        self.C.append(self.generateMonotonic(self.gamma[0], self.S_WINDOW_SIZE[0], 1, True))
        for i in range(1,self.NUM_LAYERS):
            self.C.append(self.generateMonotonic(self.gamma[i], self.S_WINDOW_SIZE[i], self.PLANES_PER_LAYER[i-1], True))

    def generateD(self):
        self.D = []
        for i in range(self.NUM_LAYERS):
            self.D.append(self.generateMonotonic(self.delta[i], self.C_WINDOW_SIZE[i], self.PLANES_PER_LAYER[i], False))
            for w in range(self.D[i].shape[0]):
                self.D[i][w] = self.D[i][w] * self.delta_bar[i]

    def distance(self, a, b):
        d = 0
        d += pow(a[0] - b[0], 2)
        d += pow(a[1] - b[1], 2)
        d = math.sqrt(d)
        return d
    
    def generateExponentialMonotonic(self, base, size, planes, norm):
        output = np.empty((pow(size, 2)))
        center = (float(size) - 1) / 2
        center = (center, center)
        index = 0
        for x in range(size):
            for y in range(size):
                distance = self.distance(center, (x, y))
                output[index] = np.exp(-base * distance)
                index += 1
        if norm:
            total = np.sum(output)
            mult = float(1) / (planes * total)
            output *= mult
        return output
    
    #Definition of the monotonic decreasing function
    def generateMonotonic(self, base, size, planes, norm):
        output = np.empty((pow(size, 2)))
        center = (float(size) - 1)/2
        center = (center, center)
        index = 0
        for x in range(size):
            for y in range(size):
                output[index] = pow(base, self.distance(center, (x,y)))
                index += 1
        if norm:
            total = 0
            total = np.sum(output)
            mult = float(1) / (planes * total)
            output *= mult
            
        return output
    
    def plotMonotonic(self, output, size):
        output_reshaped = output.reshape((size, size))
        plt.imshow(output_reshaped, cmap='viridis', origin='lower')
        plt.colorbar(label='Value')
        plt.title('Monotonic Function Output')
        plt.show()

    def pickle(self, fileName):
        pk.dump(self, open(fileName, 'wb'))

    def loadPickle(self, fileName):
        return pk.load(open(fileName, 'rb'))