import idx2numpy
import cv2 as cv
import numpy as np

img_file = 'train-images.idx3-ubyte'
label_file = 'train-labels.idx1-ubyte'
arr_img = idx2numpy.convert_from_file(img_file)
arr_labels = idx2numpy.convert_from_file(label_file)
# arr is now a np.ndarray type of object of shape 60000, 28, 28

def shuffle(arr_img, arr_labels):
    l = []
    for i in range(len(arr_img)):
        l.append((arr_img[i], arr_labels[i]))
    np.random.shuffle(l)
    return l


def numzeros(fileNum):
		if fileNum != 0:	
			if int((fileNum+1)/10) == 0:
				return 2
			else:
				return 1
		else:
			return 2
cap = 50
count = 0
i = 0
l = shuffle(arr_img, arr_labels)
while count < cap:
    image, label = l[i]
    if label == 4:
        index = numzeros(count)
        image_filename = f'train/{label}/' + str(label) + '-' + '0'*index + str(count + 1) + '.png'
        resized_image = cv.resize(image, (16, 16))
        cv.imwrite(image_filename, resized_image)
        count += 1
    i+=1
