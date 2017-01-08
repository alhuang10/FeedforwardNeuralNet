import numpy as np
import math
import random
import operator
from plotBoundary import *
import pylab as pl

def normalize_mnist(x):
	return (2.0*x / 255) - 1

# Transforms for hidden units
def rectified_linear_unit(x):
	return max(0, x)


def rectified_linear_unit_derivative(x):
	if x <= 0:
		return 0
	else:
		return 1

# Output transform
def softmax(neural_net_output):

	exponential_output = [math.exp(x) for x in neural_net_output]
	assert(len(exponential_output) == len(neural_net_output))

	# Normalize the exponential vector to get valid probability output
	softmax_output = [x/sum(exponential_output) for x in exponential_output]

	# print "Softmax output:" 
	# print np.transpose(np.matrix(softmax_output)), np.transpose(np.matrix(softmax_output)).shape

	return np.transpose(np.matrix(softmax_output))

class Neural_Network():


	# hidden_layers - specifies in order the number of nodes in each hidden layer, length of the list refers to the number of hidden layers
	# num_outputs - number of output nodes, one corresponding to each class in softmax

	def __init__(self,num_inputs,num_outputs, hidden_layers=[], learning_rate=0.005):

		
			
		self.weight_matrices = [] # Instantiate weights, each weight matrix will have dimension (number of nodes in current layer BY number of nodes in previous layer 
			# num_inputs is the first layer count, then all the hidden layers counts, then num_outputs for final layer count
			# Corresponds to index in hidden_layers, with last entry corresponding to output layer
		self.bias_vectors = [None] * (len(hidden_layers) + 1) # bias vector for each of the hidden layers and the output layer

		# Activation_vectors and weight_inputs are recalculated for each training and prediction point
		# activation vectors are the non-linear transforms of the weighted inputs (length n vector for each layer of n nodes)
		self.activation_vectors = [None] * (len(hidden_layers) + 1) 
		self.weighted_inputs = [None] * (len(hidden_layers) + 1)

		# Learning rate for stochastic gradient descent
		self.learning_rate = learning_rate

		self.num_hidden_layers = len(hidden_layers)
		self.num_output_classes = num_outputs


		#
		# Instantiate the weight matrices for each layer using a Gaussian normal with variance inversely proportional to number of nodes in the layer
		#
		for i in range(len(hidden_layers)):

			prev_layer_count = 0
			curr_layer_count = 0
			if i == 0: # previous layer for first hidden layer is the input layer
				prev_layer_count = num_inputs
			else:
				prev_layer_count = hidden_layers[i-1]
			curr_layer_count = hidden_layers[i]

			# Should instantiate weights to have mean of 0 and standard deviation of 1/sqrt(curr_layer_count) with weights matrix of [curr_layer_count, prev_layer_count]

			weight_matrix = np.matrix(np.random.normal(loc=0.0, scale=1.0/math.sqrt(curr_layer_count), size=(curr_layer_count, prev_layer_count)))

			self.weight_matrices.append(weight_matrix)

		# Instantiate weight_matrix for output layer:
		prev_layer_count = hidden_layers[-1]
		curr_layer_count = num_outputs

		output_weight_matrix = np.matrix(np.random.normal(loc=0.0, scale=1.0/math.sqrt(curr_layer_count), size=(curr_layer_count, prev_layer_count)))
		self.weight_matrices.append(output_weight_matrix)

		# Instantiate bias vectors to be 0, each hidden layer then output layer, this and activations must be [m by 1]
			# because weights matrix will multiply the activations ([n by m] * [m by 1] = [n by 1]), which works out for the next layer
		for i in range(len(hidden_layers)):
			self.bias_vectors[i] = np.matrix(np.zeros(hidden_layers[i])).transpose()
		self.bias_vectors[-1] = np.matrix(np.zeros(num_outputs)).transpose()
		

	# Pass in points stochastically to train weights via backpropagation
	# 1. Feedforward to calculate weighted_inputs and activation_values for each layer (one value for each node in layer)
	# 2. Calculate the output error
	# 3. Backpropagate to get the error for each layer
	# 4. Use that error to compute the gradient for each weight and each bias
	def train(self, point):

		input_value = point[0]
		output_value = int(point[1]) # Use output_value to index a list

		# Vectorize hidden unit non-linear tranforms
		vec_relu = np.vectorize(rectified_linear_unit)
		vec_relu_derivative = np.vectorize(rectified_linear_unit_derivative)

		neural_input = np.matrix(input_value).transpose()
		for i in range(len(self.weight_matrices)-1): # Compute only for hidden layers

			#print "Weight matrix for layer: ", i
			#print self.weight_matrices[i], self.weight_matrices[i].shape
			#print "Output of previous: "
			#print neural_input, neural_input.shape

			weighted_input_to_layer = np.dot(self.weight_matrices[i], neural_input) + self.bias_vectors[i]
			self.weighted_inputs[i] = weighted_input_to_layer

			activated_vector = vec_relu(weighted_input_to_layer)
			self.activation_vectors[i] = activated_vector

			# Update the input to the next layer as what the current layer outputs are
			neural_input = activated_vector

		# Output (neural input will be the outputs of the last hidden layer)
		output_layer_weighted_inputs = np.dot(self.weight_matrices[-1], neural_input) + self.bias_vectors[-1]
		self.weighted_inputs[-1] = output_layer_weighted_inputs
		self.activation_vectors[-1] = softmax(output_layer_weighted_inputs)
		# activation_vectors[-1] now holds the softmax outputs

		# Convert output_value to a one-hot vector
		one_hot = [0.0] * self.num_output_classes
		one_hot[output_value] = 1.0
		one_hot_matrix = np.transpose(np.matrix(one_hot))

		# Output error for softmax and cross entropy
		output_error = self.activation_vectors[-1] - one_hot_matrix

		# error_layers starts with output error which is used to backpropagate and fill in errors for each of the hidden layers
		error_layers = [None] * (self.num_hidden_layers+1)
		error_layers[-1] = output_error
		# Populate the error layers from back to start (self.num_hidden_layers-1) is the index of the last hidden layer in our other lists
		for i in range(self.num_hidden_layers-1, -1, -1):


			next_layer_weight_times_error =  np.dot(np.transpose(self.weight_matrices[i+1]), error_layers[i+1])
			weighted_input_to_layer_derivative = vec_relu_derivative(self.weighted_inputs[i])

			error_layers[i] =  np.multiply(next_layer_weight_times_error, weighted_input_to_layer_derivative) # Hadamard derivative

		# After calculating error layers, go through and adjust the weights and bias of each layer using the gradient of the cost function and learning rate
		previous_layer_activation = np.transpose(np.matrix(input_value)) # start with the activation of the input layer which is just inputs
		
		for j in range(len(self.weight_matrices)):
			#print j, previous_layer_activation

			weight_matrix_gradient = np.dot(error_layers[j], np.transpose(previous_layer_activation))
			assert self.weight_matrices[j].shape == weight_matrix_gradient.shape
			self.weight_matrices[j] = self.weight_matrices[j] - self.learning_rate * weight_matrix_gradient

			bias_matrix_gradient = error_layers[j]
			assert self.bias_vectors[j].shape == bias_matrix_gradient.shape
			self.bias_vectors[j] = self.bias_vectors[j] - self.learning_rate * bias_matrix_gradient

			previous_layer_activation = self.activation_vectors[i]



	def predict(self, point):

		input_value = point

		# Vectorize hidden unit non-linear tranforms
		vec_relu = np.vectorize(rectified_linear_unit)
		vec_relu_derivative = np.vectorize(rectified_linear_unit_derivative)

		neural_input = np.matrix(input_value).transpose()
		for i in range(len(self.weight_matrices)-1): # Compute only for hidden layers

			weighted_input_to_layer = np.dot(self.weight_matrices[i], neural_input) + self.bias_vectors[i]
			self.weighted_inputs[i] = weighted_input_to_layer

			activated_vector = vec_relu(weighted_input_to_layer)
			self.activation_vectors[i] = activated_vector

			# Update the input to the next layer as what the current layer outputs are
			neural_input = activated_vector

		# Output (neural input will be the outputs of the last hidden layer)
		output_layer_weighted_inputs = np.dot(self.weight_matrices[-1], neural_input) + self.bias_vectors[-1]
		self.weighted_inputs[-1] = output_layer_weighted_inputs
		self.activation_vectors[-1] = softmax(output_layer_weighted_inputs)

		softmax_values_list = self.activation_vectors[-1].A1
		
		# Get the class with the highest softmax value
		index, value = max(enumerate(softmax_values_list), key=operator.itemgetter(1))
	 	return index


def parse_data_file(filename):

	with open(filename) as f:
		a = f.read().split('\n')

	data_points = []

	for line in a:
		if len(line) > 0:
			numbers = map(float, line.split(' '))
			data_points.append((numbers[:-1], max(numbers[-1], 0.0)))

	return data_points

if __name__ == '__main__':


	# Sections
		# Digit Testing Input
		# Data Input
		# Neural Network Instantiation
		# Training
		# Results


	### Digit Testing ### 

	# data_digit_a = loadtxt('datasets/mnist_digit_0.csv')
	# train_a = data_digit_a[0:200,:]
	# validate_a = data_digit_a[200:350,:]
	# test_a = data_digit_a[350:500,:]

	# data_digit_b = loadtxt('datasets/mnist_digit_1.csv')
	# train_b = data_digit_b[0:200,:]
	# validate_b = data_digit_b[200:350,:]
	# test_b = data_digit_b[350:500,:]

	# data_digit_c = loadtxt('datasets/mnist_digit_2.csv')
	# train_c = data_digit_c[0:200,:]
	# validate_c = data_digit_c[200:350,:]
	# test_c = data_digit_c[350:500,:]

	# data_digit_d = loadtxt('datasets/mnist_digit_3.csv')
	# train_d = data_digit_d[0:200,:]
	# validate_d = data_digit_d[200:350,:]
	# test_d = data_digit_d[350:500,:]

	# data_digit_e = loadtxt('datasets/mnist_digit_4.csv')
	# train_e = data_digit_e[0:200,:]
	# validate_e = data_digit_e[200:350,:]
	# test_e = data_digit_e[350:500,:]

	# data_digit_f = loadtxt('datasets/mnist_digit_5.csv')
	# train_f = data_digit_f[0:200,:]
	# validate_f = data_digit_f[200:350,:]
	# test_f = data_digit_f[350:500,:]

	# data_digit_g = loadtxt('datasets/mnist_digit_6.csv')
	# train_g = data_digit_g[0:200,:]
	# validate_g = data_digit_g[200:350,:]
	# test_g = data_digit_g[350:500,:]

	# data_digit_h = loadtxt('datasets/mnist_digit_7.csv')
	# train_h = data_digit_h[0:200,:]
	# validate_h = data_digit_h[200:350,:]
	# test_h = data_digit_h[350:500,:]

	# data_digit_i = loadtxt('datasets/mnist_digit_8.csv')
	# train_i = data_digit_i[0:200,:]
	# validate_i = data_digit_i[200:350,:]
	# test_i = data_digit_i[350:500,:]

	# data_digit_j = loadtxt('datasets/mnist_digit_9.csv')
	# train_j = data_digit_j[0:200,:]
	# validate_j = data_digit_j[200:350,:]
	# test_j = data_digit_j[350:500,:]

	# X_train = np.concatenate((train_a, train_b, train_c, train_d, train_e, train_f, train_g, train_h, train_i, train_j), axis=0)
	# X_train = normalize_mnist(X_train)

	# Y0 = np.zeros((200, 1))
	# Y1 = np.zeros((200, 1))
	# Y1.fill(1)
	# Y2 = np.zeros((200, 1))
	# Y2.fill(2)
	# Y3 = np.zeros((200, 1))
	# Y3.fill(3)
	# Y4 = np.zeros((200, 1))
	# Y4.fill(4)
	# Y5 = np.zeros((200, 1))
	# Y5.fill(5)
	# Y6 = np.zeros((200, 1))
	# Y6.fill(6)
	# Y7 = np.zeros((200, 1))
	# Y7.fill(7)
	# Y8 = np.zeros((200, 1))
	# Y8.fill(8)
	# Y9 = np.zeros((200, 1))
	# Y9.fill(9)

	# Y_train = np.concatenate((Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9), axis=0)

	# Y_train_values = [a[0] for a in Y_train]

	# training_points = zip(X_train, Y_train_values)


	# X_validate = np.concatenate((validate_a, validate_b, validate_c, validate_d, validate_e, validate_f, validate_g, validate_h, validate_i, validate_j), axis=0)
	# X_validate = normalize_mnist(X_validate)

	# Y0 = np.zeros((150, 1))
	# Y1 = np.zeros((150, 1))
	# Y1.fill(1)
	# Y2 = np.zeros((150, 1))
	# Y2.fill(2)
	# Y3 = np.zeros((150, 1))
	# Y3.fill(3)
	# Y4 = np.zeros((150, 1))
	# Y4.fill(4)
	# Y5 = np.zeros((150, 1))
	# Y5.fill(5)
	# Y6 = np.zeros((150, 1))
	# Y6.fill(6)
	# Y7 = np.zeros((150, 1))
	# Y7.fill(7)
	# Y8 = np.zeros((150, 1))
	# Y8.fill(8)
	# Y9 = np.zeros((150, 1))
	# Y9.fill(9)

	# Y_validate = np.concatenate((Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9), axis=0)
	# Y_validate_values = [a[0] for a in Y_validate]
	# validation_points = zip(X_validate, Y_validate_values)


	# X_test = np.concatenate((test_a, test_b, test_c, test_d, test_e, test_f, test_g, test_h, test_i, test_j), axis=0)
	# X_test = normalize_mnist(X_test)

	# Y0 = np.zeros((150, 1))
	# Y1 = np.zeros((150, 1))
	# Y1.fill(1)
	# Y2 = np.zeros((150, 1))
	# Y2.fill(2)
	# Y3 = np.zeros((150, 1))
	# Y3.fill(3)
	# Y4 = np.zeros((150, 1))
	# Y4.fill(4)
	# Y5 = np.zeros((150, 1))
	# Y5.fill(5)
	# Y6 = np.zeros((150, 1))
	# Y6.fill(6)
	# Y7 = np.zeros((150, 1))
	# Y7.fill(7)
	# Y8 = np.zeros((150, 1))
	# Y8.fill(8)
	# Y9 = np.zeros((150, 1))
	# Y9.fill(9)

	# Y_test = np.concatenate((Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9), axis=0)
	# Y_test_values = [a[0] for a in Y_test]
	# test_points = zip(X_test, Y_test_values)

	### End of Digit data input



	### Data Input ###

	data_set_number = '2'
	training_file = "datasets/data" + data_set_number + "_train.csv"
	validation_file = "datasets/data" + data_set_number + "_validate.csv"
	testing_file = "datasets/data" + data_set_number + "_test.csv"

	training_points = parse_data_file(training_file)
	validation_points = parse_data_file(validation_file)
	test_points = parse_data_file(testing_file)



	### Neural network instantiation ###

	# num_inputs = 784
	# num_output_classes = 10
	# hidden_layer_architecture = [500] 
	num_inputs = 2
	num_output_classes = 2
	# hidden_layer_architecture = [10]
	# architecture_description = "One Small Layer"
	# hidden_layer_architecture = [10, 10]
	# architecture_description = "Two Small Layers"
	# hidden_layer_architecture = [200]
	# architecture_description = "One Large Layer"
	hidden_layer_architecture = [200,200]
	architecture_description = "Two Large Layers"

	neural_net_instance = Neural_Network(num_inputs, num_output_classes, hidden_layer_architecture, learning_rate=0.005)

	

	### Training ###

	number_of_training_iterations = 100000

	print ""
	print "Training neural net on", number_of_training_iterations,"iterations"
	print "Number of hidden layers: ", len(hidden_layer_architecture)
	print "Hidden layer architecture: ", hidden_layer_architecture
	print ""

	check_after_num_iterations = 2000
	previous_checked_validation_accuracy = 0.0

	for i in range(number_of_training_iterations):


		# Pick a random point to train on
		index = random.randint(0, len(training_points)-1)
		training_point = training_points[index]

		neural_net_instance.train(training_point)

		## Calculating validation set error to see when to stop
		if i%check_after_num_iterations==0:
			print "Currently on iteration ", i

			incorrect_count = 0.0
			for v_point in validation_points:
				predicted_value = neural_net_instance.predict(v_point[0])
				if predicted_value != int(v_point[1]):
					incorrect_count += 1.0
			current_validation_loss = incorrect_count / len(validation_points)

			# print "Current validation accuracy: ", current_validation_loss
			# print "Previous validation accuracy: ", previous_checked_validation_accuracy

			if current_validation_loss == previous_checked_validation_accuracy:
				print "Iterations for convergence: ", i, current_validation_loss
				break

			previous_checked_validation_accuracy = current_validation_loss

	### For plotting test data sets

	train = loadtxt('./datasets/data'+data_set_number+'_train.csv')
	X = train[:,0:2]
	Y = train[:,2:3]

	plotDecisionBoundary(X, Y, neural_net_instance.predict, [0.5], title = 'Neural Net Training Set ' + data_set_number + " - " + architecture_description)
	pl.show()

	test = loadtxt('./datasets/data'+data_set_number+'_test.csv')
	X = test[:,0:2]
	Y = test[:,2:3]

	plotDecisionBoundary(X, Y, neural_net_instance.predict, [0.5], title = 'Neural Net Test Set ' + data_set_number + " - " + architecture_description)
	pl.show()



	### Results ###

	## Training set
	correct_count = 0.0
	incorrect_count = 0.0
	for i in range(len(training_points)):

		index = i
		test_point = training_points[index]

		predicted_value = neural_net_instance.predict(test_point[0])

		if predicted_value == int(test_point[1]):
			#print "Correct"
			correct_count += 1.0
		else:
			#print "Wrong"
			incorrect_count += 1.0


	print "Training:", len(training_points), "points"
	print "Correct fraction (Training):", correct_count / (correct_count + incorrect_count), ", Correct count: ", correct_count
	print "Incorrect fraction (Training):", incorrect_count / (correct_count + incorrect_count), ", Incorrect count: ", incorrect_count
	print ""


	### Validation set
	correct_count = 0.0
	incorrect_count = 0.0
	for i in range(len(validation_points)):

		index = i
		test_point = validation_points[index]

		predicted_value = neural_net_instance.predict(test_point[0])

		if predicted_value == int(test_point[1]):
			#print "Correct"
			correct_count += 1.0
		else:
			#print "Wrong"
			incorrect_count += 1.0


	print "Validation:", len(validation_points), "points"
	print "Correct fraction (Validation):", correct_count / (correct_count + incorrect_count), ", Correct count: ", correct_count
	print "Incorrect fraction (Validation):", incorrect_count / (correct_count + incorrect_count), ", Incorrect count: ", incorrect_count
	print ""


	### Test set
	correct_count = 0.0
	incorrect_count = 0.0
	for i in range(len(test_points)):

		index = i
		test_point = test_points[index]

		predicted_value = neural_net_instance.predict(test_point[0])

		if predicted_value == int(test_point[1]):
			#print "Correct"
			correct_count += 1.0
		else:
			#print "Wrong"
			incorrect_count += 1.0


	print "Test:", len(test_points), "points"
	print "Correct fraction (Test):", correct_count / (correct_count + incorrect_count), ", Correct count: ", correct_count
	print "Incorrect fraction (Test):", incorrect_count / (correct_count + incorrect_count), ", Incorrect count: ", incorrect_count
	print ""


