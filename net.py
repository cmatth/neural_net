import random
import data
import node
#import sklearn.linear_model.LogisticRegression.predict_proba as softmax
import numpy as np

training_path = '/home/kmoney/Documents/neural_net/training.txt'
testing_path = '/home/kmoney/Documents/neural_net/testing.txt'
datas = data.readFromFile(training_path)
trainingSet = data.readIntoClasses(datas)
datas = data.readFromFile(testing_path)
testingSet = data.readIntoClasses(datas)

'''
for x in trainingSet:
	print x.label
	print x.data
'''
data.normalizeData(trainingSet)
data.normalizeData(testingSet)

learningRate = 1

network = []
inputLayer = []
hiddenLayer = []
outputLayer = []

for x in range(0,63): inputLayer.append(node.neuron(learningRate,True,False))
for x in range(0,63): hiddenLayer.append(node.neuron(learningRate,False, False))
for x in range(0,7): outputLayer.append(node.neuron(learningRate,False,True))

network.append(inputLayer)
network.append(hiddenLayer)
network.append(outputLayer)

expected = data.classOutputs()

#print len(trainingSet[0].data)


'''
for x in trainingSet:
	for y in range(len(inputLayer)):
		#print x.data
		#inputs = []
		#inputs.append(x.data[y])
		print "input", x.data[y]
		inputLayer[y].activation(x.data[y])
		outtie = inputLayer[y].output
		#outputs.append(inputLayer[y].output)
		print "output:",outtie
		#print len(outputs)
	raw_input("PRESS ENTER TO END")
'''

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def runDatum(network, datum):
	outputs = []
	print "input layer:"
	#propogate datum through input layer
	for y in range(len(network[0])):
		network[0][y].activation(datum.data[y])
		outputs.append(network[0][y].output)
	raw_input("hidden layer")
	#propogate datum through the hidden layer
	outputs1 = []
	for x in range(0,len(network[1])):
		network[1][x].activation(outputs)
		outputs1.append(network[1][x].output)

	raw_input("output layer")
	#propogate datum to output layer
	finalOut = []
	for x in range(0,len(network[2])):
		network[2][x].activation(outputs1)
		finalOut.append(network[2][x].output)

	raw_input('next example')
	#print finalOut
	finalOut = softmax(finalOut)
	print finalOut
	#print sum(finalOut)

def backPropagate(network, trueOut):
	# get errors in output layer
	for x in range(0,len(network[2])):
		network[2][x].calculateError(trueOut[x])

	# get errors in hidden layer
	node.accumlateError(network[1], network[2])

	#adjust weights for all nodes in network
	for x in network[2]:
		x.adjustWeight()
		#reset input
		x.input = 0
		x.output = 0

for x in trainingSet:
	runDatum(network, x)
	#backPropagate(network, expected[x.label])







