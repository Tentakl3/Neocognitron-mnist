import numpy as np
import location

class Message(object):
    def __init__(self, planes, initSize):
        self.numPlanes = planes
        self.size = initSize
        self.outputs = np.zeros((self.numPlanes, self.size, self.size))

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
        startX = x - windowSize//2
        startY = y - windowSize//2
        count = 0
        for i in range(windowSize):
            for j in range(windowSize):
                if startX + i >= 0 and startY + j >= 0 and startX + i < self.size and startY + j < self.size:
                    output[count] = self.outputs[plane][startX + i][startY + j]
                count += 1
        return output
    
    def getSquareWindows(self, x, y, windowSize):
        output = np.zeros((self.numPlanes, windowSize, windowSize))
        for plane in range(self.numPlanes):
            output[plane] = self.getSquareWindow(plane, x, y, windowSize)
        return output

    def getSquareWindow(self, plane, x, y, windowSize):
        output = np.zeros((windowSize, windowSize))
        offset = (windowSize-1)//2
        #offset = windowSize//2
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
        for plane in range(sColumn.shape[0]):
            for x in range(sColumn.shape[1]):
                for y in range(sColumn.shape[2]):
                    if sColumn[plane][x][y] > maxVal:
                        maxL = location.Location(plane, x, y)
                        maxVal = sColumn[plane][x][y]
        offset = (windowSize - windowSize%2)//2
        #offset = windowSize // 2
        if maxL is not None:
            x, y = maxL.getPoint()
            maxL.setPoint(x+center[0]-offset, y+center[1]-offset)
        return maxL
    
    def getSingleOutput(self, loc):
        if location is None:
            return 0
        plane = loc.getPlane()
        x, y = loc.getPoint()
        return self.outputs[plane][x][y]

    def getMaxPerPlane(self, plane, points):
        p = None
        maxVal = 0.0
        for point in points:
            temp = point
            if temp is None:
                p = None
            elif temp.getPlane() == plane:
                if self.getSingleOutput(temp) > maxVal:
                    maxVal = self.getSingleOutput(temp)
                    p = temp.getPoint()
        return p

    def getRepresentatives(self, columnSize):
        points = []
        offset = (columnSize - 1) // 2
        #offset = columnSize // 2
        if columnSize == self.size:
            sColumn = self.getSquareWindows(self.size//2, self.size//2, columnSize)
            temp = self.getLocationOfMax(sColumn, (self.size//2, self.size//2), columnSize)
            points.append(temp)
        else:
            for x in range(self.size-offset):
                for y in range(self.size-offset):
                    sColumn = self.getSquareWindows(x, y, columnSize)
                    temp = self.getLocationOfMax(sColumn, (x, y), columnSize)
                    points.append(temp)

        reps = []
        for plane in range(self.numPlanes):
            reps.append(self.getMaxPerPlane(plane, points))
        return reps




