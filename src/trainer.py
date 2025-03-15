import random
import cv2 as cv
import neocognitron
import initStruct
import os

IMG_SIZE = 16
FILES_PER_CLASS = 3
TRAIN_PER_CLASS = 20
ALPHABET = '012'
DATA_DIR = 'data/'
TRAIN_DATA_DIR = 'data/training/'
ON = 0.
OFF = 255.

def train(init):
    network = neocognitron.Neocognitron(init)
    trainTemplates = []
    trainTemplates = getTrainFile()
    count = 0
    for letter in trainTemplates:
        print(f'sample {count}')
        network.train_propagate(letter)
        count+=1
    """
	for layer in range(init.NUM_LAYERS):
		trainTemplates = []
		for plane in range(init.PLANES_PER_LAYER[layer]):
			trainTemplates.append(getTrainFile(init, layer, plane))
		print("Training layer" + str(layer + 1))
		network.trainLayer(layer, trainTemplates)
	"""

    return network

def getTrainFile():
    output = []
    for stimuli in ALPHABET:
        path = TRAIN_DATA_DIR + str(stimuli) + '/'
        for folder, subfolders, contents in os.walk(path):
            for content in contents:
                if not content[0] == '.':
                    img = cv.imread(path + content, flags=cv.IMREAD_GRAYSCALE)
                    for x in range(img.shape[0]):
                        for y in range(img.shape[1]):
                            if img[x][y] == OFF: img[x][y] = ON
                            elif img[x][y] == ON: img[x][y] = 1
                output.append(img)
    random.shuffle(output)
    return output

def numzeros(fileNum):
		if fileNum != 0:	
			if int((fileNum+1)/10) == 0:
				return 2
			else:
				return 1
		else:
			return 2
		
def getInputs(trainFiles):
	inputs = []
	for letter in ALPHABET:
		for fileNum in trainFiles:
			numZeros = numzeros(fileNum)
			fileName = letter + '-' + '0'*numZeros + str(fileNum + 1) + '.png'						
			img = cv.imread(DATA_DIR + letter+ '/' + fileName, flags=cv.IMREAD_GRAYSCALE)
			for x in range(img.shape[0]):
					for y in range(img.shape[1]):
						if img[x][y] == OFF:
							img[x][y] = ON
						elif img[x][y] == ON: 
							img[x][y] = 1
			inputs.append((img, letter))
	random.shuffle(inputs)
	return inputs

def validate(network):
	numCorrect = 0
	numTotal = 0
	#range(FILES_PER_CLASS)
	validateInputs = getInputs(range(FILES_PER_CLASS))
	print ('TESTING')
	#len(validateInputs)
	network.setDictionary()
	for n in range(len(validateInputs)):
		print ('TESTING LETTER ' + validateInputs[n][1])
		guess = network.propagate(validateInputs[n][0], validateInputs[n][1])
                  
def runTraining():
	init = initStruct.InitStruct()
	network = train(init)
	#save network ?
	return network