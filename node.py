import random
from math import exp

class NeuralNet:
	def __init__(self, numInputNeurons, numHiddenNeurons, numOutputNeurons, learningRate):
		self.learnRate 	 = learningRate
		self.learnStrat  = 'online'
		self.finalOut 	 = []

		self.inputLayer  = Layer('input',numInputNeurons,0).neurons
		self.hiddenLayer = Layer('hidden',numHiddenNeurons,numInputNeurons).neurons
		self.outputLayer = Layer('output',numOutputNeurons,numHiddenNeurons).neurons


	def forwardPropagate(self, datum):
		self.inputOut  = []
		self.hiddenOut = []
		self.finalOut  = []
		# propagate datum through input layer
		for y in range(0, len(self.inputLayer) - 1):
			self.inputLayer[y].activation(datum.data[y])
			self.inputOut.append(self.inputLayer[y].output)
		# bias node
		self.inputLayer[-1].activation(1)
		self.inputOut.append(self.inputLayer[-1].output)

		# propogate datum through the hidden layer
		for x in range(0, len(self.hiddenLayer) - 1):
			self.hiddenLayer[x].activation(outputs)
			self.hiddenOut.append(self.hiddenLayer[x].output)
		# bias node
		self.hiddenLayer[-1].activation([1])
		self.hiddenOut.append(self.hiddenLayer[-1].output)

		# propagate datum to output layer
		for x in range(0, len(self.outputLayer)):
			self.outputLayer[x].activation(outputs1)
			self.finalOut.append(self.outputLayer[x].output)
		normalize(self.finalOut)


	def backPropagate(self, trueOut):
		# get errors in output layer
		for x in range(0, len(self.outputLayer)):
			self.outputLayer[x].calculateError(trueOut[x], self.finalOut[x])
			self.outputLayer[x].calculateDeltas(hiddenOut)

		# get errors in hidden layer
		accumlateError(self.hiddenLayer, self.outputLayer)
		for x in network[1]: x.calculateDeltas(self.inputOut)

		if self.learnStrat == 'online':
			# adjust weights for all nodes in network
			for x in network[2]: x.adjustWeights()
			for x in network[1]: x.adjustWeights()

class Layer(NeuralNet):
	def __init__(self, type, numNeurons, numWeights):
		self.neurons = []
		output = False
		input  = False
		if   type == 'input': input   = True
		elif type == 'output': output = True
		for x in range(0,numNeurons):
			layer.append(Neuron(self.learnRate,input,output,False,numWeights))
		layer.append(Neuron(self.learnRate,input,output,True,numWeights))


class Neuron(Layer):
	def __init__(self, Lrate, inputBool, outputBool, biasBool, numWeights):
		self.weights = []
		self.delta =[]
		self.output = 0
		self.error = 0
		self.input = 0
		self.learnRate = Lrate
		self.tDer = 0
		self.inputNode = inputBool
		self.outputNode = outputBool
		self.bias = biasBool

		for x in range(0, numWeights):
			if not inputBool: self.weights.append(random.uniform(-.25, .25))
			else: self.weights.append(1)

	def activation(self, input):
			#calculate activation value
			if self.inputNode:
				self.input = input
			else:
				self.input = 0
				for x in range(0,len(input)):
					#print "input:", input[x], "weight", self.weights[x]
					self.input += input[x] * self.weights[x]

			#apply to activation function
			if self.inputNode:
				self.output = self.input
			else:
				param = 1.1
				self.output =  (1.0 - exp(-self.input * param)) / (1.0 + exp(-self.input * param))
			#calculate transfer derivative
			#self.tDer = self.output * (1.0 - self.output)
			self.tDer = (self.input / 2) * (1 + self.output) * (1 - self.output)

			if self.bias:
				self.input = 1
				self.output = 1


	def calculateError(self, expected, actual):
			if self.outputNode:
				self.tDer = actual * (1.0 - actual)
				#print "*", expected, actual
				self.error = (expected - actual) * self.tDer
				#print self.error

	def calculateDeltas(self, inputs):
		self.delta = []
		for x in range(0,len(inputs)):
			self.delta.append(self.learnRate * self.error * inputs[x])

	def adjustWeights(self):
		for x in range(len(self.weights)):
			self.weights[x] = self.weights[x] + self.delta[x]

def accumlateError(layer, successor):
	for i in range(0,len(layer)):
		for x in successor:
			layer[i].error += x.error * x.weights[i]
		layer[i].error = layer[i].error * layer[i].tDer

def getLayerErrors(layer):
	errors = []
	for neuron in layer:
		errors.append(neuron.error)
	return errors

def normalize(finalOut):
    mean = sum(finalOut) / 7
    Rnge = max(finalOut) - min(finalOut) + .0001
    for x in range(0,len(finalOut)):
        finalOut[x] = (finalOut[x] - mean) / Rnge
    return finalOut

