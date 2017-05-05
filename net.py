import data
import node
import numpy as np

#training_path = '/home/kmoney/Documents/neural_net/training.txt'
#testing_path = '/home/kmoney/Documents/neural_net/testing.txt'
training_path = '/home/casey/PycharmProjects/neural_net/training.txt'
testing_path = '/home/casey/PycharmProjects/neural_net/testing.txt'
trainingSet, testingSet = data.getDataSets(training_path, testing_path)


####################################################################
learningRate = .15
inputNodes = len(testingSet[0].data)
hiddenNodes = 250
####################################################################


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def normalize(finalOut):
    mean = sum(finalOut) / 7
    Rnge = max(finalOut) - min(finalOut) + .0001
    for x in range(0,len(finalOut)):
        finalOut[x] = (finalOut[x] - mean) / Rnge
    return finalOut

def runDatum(network, datum):
    outputs = []
    #propogate datum through input layer
    for y in range(0,len(network[0])-1):
        network[0][y].activation(datum.data[y])
        outputs.append(network[0][y].output)
    #bias node
    network[0][-1].activation(1)
    outputs.append(network[0][-1].output)

    #propogate datum through the hidden layer
    outputs1 = []
    for x in range(0,len(network[1])-1):
        network[1][x].activation(outputs)
        outputs1.append(network[1][x].output)
    #bias node
    network[1][-1].activation([1])
    outputs1.append(network[1][-1].output)

    #propogate datum to output layer
    finalOut = []
    for x in range(0,len(network[2])):
        network[2][x].activation(outputs1)
        finalOut.append(network[2][x].output)

    normalize(finalOut)
    return finalOut, outputs1, outputs

def backPropagate(network, trueOut, finalOut, hiddenOut, inputOut):
    # get errors in output layer
    for x in range(0,len(network[2])):
        network[2][x].calculateError(trueOut[x], finalOut[x])
        network[2][x].calculateDeltas(hiddenOut)

    # get errors in hidden layer
    node.accumlateError(network[1], network[2])
    for x in network[1]: x.calculateDeltas(inputOut)

    #adjust weights for all nodes in network
    for x in network[2]: x.adjustWeights()
    for x in network[1]: x.adjustWeights()

for z in range(0,1):
    avg = 0
    for y in range(0,400):

        network = []
        inputLayer = []
        hiddenLayer = []
        outputLayer = []
        for x in range(0, inputNodes): inputLayer.append(node.neuron(learningRate, True, False, False, 0))
        for x in range(0, hiddenNodes): hiddenLayer.append(node.neuron(learningRate, False, False, False, 64))
        for x in range(0, 7): outputLayer.append(node.neuron(learningRate, False, True, False, hiddenNodes + 1))

        # add bias nodes to input and hidden layers
        inputLayer.append(node.neuron(learningRate, True, False, True, 0))
        hiddenLayer.append(node.neuron(learningRate, False, False, True, 64))

        network.append(inputLayer)
        network.append(hiddenLayer)
        network.append(outputLayer)

        expected = data.classOutputs()
        indices = data.labelIndex()

        count = 0
        for x in trainingSet:
            count += 1
            exp = expected[x.label]
            #print exp
            guess, hiddenOut, inputOut = runDatum(network, x)
            #print x.label, " : ", indices[np.argmax(guess)]
            #for y in range(0,len(guess)):
            #    print guess[y]
                #print exp[y] - guess[y]

            backPropagate(network, expected[x.label], guess, hiddenOut, inputOut)
            #raw_input("next")

            if count%7 == 0:
                #print "hello"
                for x in hiddenLayer: x.learnRate = x.learnRate * .65
                for x in outputLayer: x.learnRate = x.learnRate * .65



        count = 0
        for x in testingSet:
            guess, hiddenOut, inputOut = runDatum(network, x)
            #print x.label, " : ", indices[np.argmax(softmax(guess))]
            if x.label == indices[np.argmax(softmax(guess))]:
                count += 1
        #print "accuracy: %f" %(float(count) / 23)
        avg += float(count)/7
    print "hidden nodes: ", hiddenNodes, " => ", avg / 20

























''''
for y in range(0,10):
    for x in trainingSet:
        count += 1
        exp = expected[x.label]
        print exp
        guess, hiddenOut, inputOut = runDatum(network, x)
        print x.label, " : ", indices[np.argmax(guess)]
        for y in range(0,len(guess)):
            print guess[y]
            #print exp[y] - guess[y]

        backPropagate(network, expected[x.label], guess, hiddenOut, inputOut)
        #raw_input("next")

        if count%7 == 0:
            #print 'hello'
            for x in hiddenLayer: x.learnRate = x.learnRate * .5
            for x in outputLayer: x.learnRate = x.learnRate * .5
'''







