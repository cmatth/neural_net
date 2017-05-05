import random
from math import exp

class neuron:
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
				self.output =  (1.0 - exp(-self.input)) / (1.0 + exp(-self.input))
			#calculate transfer derivative
			#self.tDer = self.output * (1.0 - self.output)
			self.tDer = (self.input / 2) * (1 + self.output) * (1 - self.output)

			if self.bias:
				self.input = 1
				self.output = 1


	def calculateError(self, expected, actual):
			if self.outputNode:
				self.tDer = actual * (1.0 - actual)
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


