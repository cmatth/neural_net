import random
import data
import node
#import sklearn.linear_model.LogisticRegression.predict_proba as softmax
import numpy as np

#training_path = '/home/kmoney/Documents/neural_net/training.txt'
#testing_path = '/home/kmoney/Documents/neural_net/testing.txt'
training_path = '/home/casey/PycharmProjects/neural_net/training.txt'
testing_path = '/home/casey/PycharmProjects/neural_net/testing.txt'
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

for x in range(0,63): inputLayer.append(node.neuron(learningRate,True,False,False))
for x in range(0,63): hiddenLayer.append(node.neuron(learningRate,False, False,False))
for x in range(0,7): outputLayer.append(node.neuron(learningRate,False,True,False))

# add bias nodes to input and hidden layers
inputLayer.append(node.neuron(learningRate,True,False,True))
hiddenLayer.append(node.neuron(learningRate,False,False,True))

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
	#print "input layer:"
	#propogate datum through input layer
	for y in range(0,len(network[0])-1):
		#print y
		network[0][y].activation(datum.data[y])
		outputs.append(network[0][y].output)
	#bias node
	network[0][-1].activation(1)
	outputs.append(network[0][-1].output)
	#raw_input("hidden layer")
	#propogate datum through the hidden layer
	outputs1 = []
	for x in range(0,len(network[1])-1):
		#print x
		network[1][x].activation(outputs)
		outputs1.append(network[1][x].output)
	#bias node
	network[1][-1].activation([1])
	outputs1.append(network[1][-1].output)

	#raw_input("output layer")
	#propogate datum to output layer
	finalOut = []
	for x in range(0,len(network[2])):
		network[2][x].activation(outputs1)
		finalOut.append(network[2][x].output)

	mean = sum(finalOut) / 7
	Rnge = max(finalOut) - min(finalOut)
	for x in range(0,len(finalOut)):
		finalOut[x] = (finalOut[x] - mean) / Rnge
	#print finalOut
	finalOut = softmax(finalOut)
	return finalOut
	#print "Guess:", expected

def backPropagate(network, trueOut):
	# get errors in output layer
	for x in range(0,len(network[2])):
		network[2][x].calculateError(trueOut[x])

	# get errors in hidden layer
	node.accumlateError(network[1], network[2])

	#adjust weights for all nodes in network
	for x in network[2]:
		x.adjustWeight()
	for x in network[1]:
		x.adjustWeight()

for x in trainingSet:
	print x.label
	exp = expected[x.label]
	guess = runDatum(network, x)
	for y in range(0,len(guess)):
		print exp[y] - guess[y]
	backPropagate(network, expected[x.label])
	raw_input("next")







