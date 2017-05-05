import random
from math import exp

class neuron:
	def __init__(self, Lrate, inputBool, outputBool, biasBool):
		self.weights = []
		self.output = 0
		self.error = 0
		self.input = 0
		self.learnRate = Lrate
		self.tDer = 0
		self.inputNode = inputBool
		self.outputNode = outputBool
		self.bias = biasBool

		for x in range(0,65):
			if not inputBool: self.weights.append(random.random())
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
					self.input = self.input
			#print 'total input', self.input
					#print "in", self.input

			#apply to activation function
			if self.inputNode:
				self.output = self.input
			if self.bias:
				self.input = 1
				self.output = 1
			else:
				self.output =  1.0 / (1.0 + exp(-self.input))
			#print "out", self.output
			#print '*******************************************'

			#calculate transfer derivative
			self.tDer = self.output * (1.0 - self.output)


	def calculateError(self, expected):
			if self.outputNode:
				self.error = (expected - self.output) * self.tDer
				#print self.error

	def adjustWeight(self):
		for x in range(len(self.weights)):
			self.weights[x] = self.weights[x] + self.learnRate * self.error * self.input

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


