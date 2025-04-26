import sLayer
import cLayer
import message
import cv2 as cv
import numpy as np

ALPHABET = '01234'
STORAGE_PATH = 'data/storage/layer'

class Neocognitron(object):
	def __init__(self, init):

		self.numLayers = init.NUM_LAYERS
		self.sLayers = []
		self.cLayers = []
		self.init = init

        #fill the layer list with Layer objects
		for layer in range(self.numLayers):
			self.sLayers.append(sLayer.SLayer(layer, init))
			self.cLayers.append(cLayer.CLayer(layer, init))
	
		self.setDictionary()
	
	def setDictionary(self):
		self.data_dic = {0:{'0':0, '1':0, '2':0, '3':0, '4':0}, 1:{'0':0, '1':0, '2':0, '3':0, '4':0}, 2:{'0':0, '1':0, '2':0, '3':0, '4':0}, 3:{'0':0, '1':0, '2':0, '3':0, '4':0}, 4:{'0':0, '1':0, '2':0, '3':0, '4':0}}
    
    #test the model
	def propagate(self, image, symbol):
		output = message.Message(1, self.init.INPUT_LAYER_SIZE)
		#set the value of the output and the resulting "image"
		output.setPlaneOutput(0, image)
		self.imgPlane(output, symbol, self.data_dic[0][symbol])
		self.data_dic[0][symbol] += 1
		for layer in range(self.numLayers):
			output = self.sLayers[layer].propagate(output, False)
			self.save_and_unify_images(layer, output, symbol, self.data_dic[layer+1][symbol], True)
			output = self.cLayers[layer].propagate(output)
			self.save_and_unify_images(layer, output, symbol, self.data_dic[layer+1][symbol], False)
			self.data_dic[layer+1][symbol] += 1

		result = output.getPointsOnPlanes(0, 0)
		return result

	def imgPlane(self, out, symbol, symbolIndex):
		filename = f'{STORAGE_PATH}{0}/{symbol}/img{symbolIndex}.png'
		array = out.outputs[0]
		# Convertir array a NumPy array
		array = np.array(array, dtype=np.float32)
		# Crear imagen binaria: valores diferentes de cero se pintan de negro
		binary_image = np.where(array > 0, 0, 255).astype(np.uint8)
		# Guardar la imagen binaria
		cv.imwrite(filename, binary_image)
	
	def save_and_unify_images(self, layer, out, symbol, symbolIndex, switch):
		images = []
		for plane in range(out.numPlanes):
			array = out.outputs[plane]
			# Convert array to NumPy array
			array = np.array(array, dtype=np.float32)
			# Create binary image: non-zero values are black, zero values are white
			#array = abs((array*255)-255)
			binary_image = np.where(array > 0, 0, 255).astype(np.uint8)
			# Save the binary image (optional, commented out)
			images.append(binary_image)
		
		# Create a rectangular layout for the images
		num_images = len(images)
		image_height, image_width = images[0].shape
		# Calculate the number of rows and columns for the rectangle
		num_columns = int(np.sqrt(num_images))  # Use square root for approximate width
		num_rows = int(num_images / num_columns)  # Calculate rows based on total images
		
		# Create a blank unified image with the calculated dimensions
		unified_image = np.full((num_rows * image_height, num_columns * image_width), 255, dtype=np.uint8)
		
		for idx, img in enumerate(images):
			row = idx // num_columns
			col = idx % num_columns
			unified_image[row * image_height:(row + 1) * image_height, col * image_width:(col + 1) * image_width] = img
		
		# Save the unified image
		if switch:
			unified_filename = f'{STORAGE_PATH}{layer+1}/{symbol}/img{symbolIndex}-S.png'
		else:
			unified_filename = f'{STORAGE_PATH}{layer+1}/{symbol}/img{symbolIndex}-C.png'
		cv.imwrite(unified_filename, unified_image)

		if layer == self.numLayers - 1:
			max_val = np.max(out.outputs)
			max_idx = np.unravel_index(np.argmax(out.outputs, axis=None), out.outputs.shape)
			max_plane, max_x, max_y = max_idx

			 # Create a unified image with only the max cell in black
			final_unified_image = np.full((num_rows * image_height, num_columns * image_width), 255, dtype=np.uint8)
			row = max_plane // num_columns
			col = max_plane % num_columns
			final_unified_image[row * image_height + max_x, col * image_width + max_y] = 0

			max_image_filename = f'{STORAGE_PATH}{layer+1}/{symbol}/max_value{symbolIndex}.png'
			cv.imwrite(max_image_filename, final_unified_image)
	
	def train_propagate(self, image):
		output = message.Message(1, self.init.INPUT_LAYER_SIZE)
		#set the value of the output and the resulting "image"
		output.setPlaneOutput(0, image)
		for layer in range(self.numLayers):
			output = self.sLayers[layer].propagate(output, True)
			output = self.cLayers[layer].propagate(output)