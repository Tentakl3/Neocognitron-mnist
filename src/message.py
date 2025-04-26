import numpy as np
import location

class Message(object):
    def __init__(self, planes, initSize):
        self.numPlanes = planes
        self.size = initSize
        self.outputs = np.zeros((self.numPlanes, self.size, self.size))

    def display(self):
        for plane in range(self.numPlanes):
            print ('PLANE: ' + str(plane+1))
            print (self.outputs[plane])
            
    def setPlaneOutput(self, plane, toSet):
        self.outputs[plane] = toSet

    def setOneOutput(self, plane, x, y, val):
        self.outputs[plane][x][y] = val

    def getWindows(self, x, y, windowSize):
        output = np.zeros((self.numPlanes, pow(windowSize, 2)))
        for plane in range(self.numPlanes):
            output[plane] = self.getOneWindow(plane, x, y, windowSize)
        return output

    def getOneWindow(self, plane, x, y, windowSize):
        output = np.zeros(pow(windowSize, 2))
        offset = (windowSize-1)//2
        count = 0
        for i in range(windowSize):
            for j in range(windowSize):
                if x - offset + i >= 0 and y - offset + j >= 0 and x - offset + i < self.size and y - offset + j < self.size:
                    output[count] = self.outputs[plane][x - offset + i][y - offset + j]
                count +=1
        return output
    
    def getSquareWindows(self, x, y, windowSize):
        output = np.zeros((self.numPlanes, windowSize, windowSize))
        for plane in range(self.numPlanes):
            output[plane] = self.getSquareWindow(plane, x, y, windowSize)
        return output

    def getSquareWindow(self, plane, x, y, windowSize):
        output = np.zeros((windowSize, windowSize))
        offset = (windowSize-1)//2
        for i in range(windowSize):
            for j in range(windowSize):
                if x - offset + i >= 0 and y - offset + j >= 0 and x - offset + i < self.size and y - offset + j < self.size:
                    output[i][j] = self.outputs[plane][x - offset + i][y - offset + j]
        return output
    
    def getPointsOnPlanes(self, x, y):
        output = []
        for plane in range(self.numPlanes):
            output.append(self.outputs[plane][x][y])
        return output

    def getLocationOfMax(self, sColumn, center, windowSize):
        maxL = None
        maxVal = 0.0
        for plane in range(self.numPlanes):
            for x in range(windowSize):
                for y in range(windowSize):
                    if sColumn[plane][x][y] > maxVal:
                        maxL = location.Location(plane, x, y)
                        maxVal = sColumn[plane][x][y]

        if windowSize % 2 == 0:
            offset = windowSize//2
        else:
            offset = (windowSize - 1)//2
        #offset = windowSize // 2
        if maxL is not None:
            x, y = maxL.getPoint()
            maxL.setPoint(x+center[0]-offset, y+center[1]-offset)
        return maxL
    
    def getSingleOutput(self, loc):
        plane = loc.getPlane()
        x, y = loc.getPoint()
        return self.outputs[plane][x][y]

    def getMaxPerPlane(self, plane, points):
        p = None
        maxVal = 0.0
        for point in points:
            temp = point
            if temp is not None and temp.getPlane() == plane:
                if self.getSingleOutput(temp) > maxVal:
                    maxVal = self.getSingleOutput(temp)
                    p = temp.getPoint()
        return p

    def getRepresentatives(self, columnSize):
        points = []
        for x in range(self.size):
            for y in range(self.size):
                sColumn = self.getSquareWindows(x, y, columnSize)
                temp = self.getLocationOfMax(sColumn, (x, y), columnSize)
                points.append(temp)
        reps = []
        for plane in range(self.numPlanes):
            reps.append(self.getMaxPerPlane(plane, points))
        print(reps)
        return reps




