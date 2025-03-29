import random
import cv2 as cv
import neocognitron
import initStruct
import os

IMG_SIZE = 16
FILES_PER_CLASS = 10
TRAIN_PER_CLASS = 20
ALPHABET = '01234'
DATA_DIR = 'data/validate-mnist/'
TRAIN_DATA_DIR = 'data/training-mnist/'
ON = 0
OFF = 255

def train(init):
	network = neocognitron.Neocognitron(init)
	trainTemplates = []
	trainTemplates = getTrainFile()
	count = 0
	for letter in trainTemplates:
		print(f'sample {count}')
		network.train_propagate(letter)
		count+=1

	return network

def getTrainFile():
	output = []
	counter = 0
	for stimuli in ALPHABET:
		path = TRAIN_DATA_DIR + str(stimuli) + '/'
		for folder, subfolders, contents in os.walk(path):
			for content in contents:
				if counter < TRAIN_PER_CLASS:
					if not content[0] == '.':
						img = cv.imread(path + content, flags=cv.IMREAD_GRAYSCALE)
						_, img = cv.threshold(img, 225, 1, cv.THRESH_BINARY_INV)
					output.append(img)
					counter += 1
		counter = 0
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
			_, img = cv.threshold(img, 225, 1, cv.THRESH_BINARY_INV)
			inputs.append((img, letter))
	random.shuffle(inputs)
	return inputs

def validate(network):
	validateInputs = getInputs(range(FILES_PER_CLASS))
	print ('TESTING')
	network.setDictionary()
	for n in range(len(validateInputs)):
		print ('TESTING LETTER ' + validateInputs[n][1])
		guess = network.propagate(validateInputs[n][0], validateInputs[n][1])
                  
def runTraining():
	init = initStruct.InitStruct()
	network = train(init)
	return network
