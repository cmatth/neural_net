import data
import node
import numpy as np
import readData as rd
import plot as pl

home = rd.homefolder()
training_path = home + 'training.txt'
testing_path =  home + 'testing.txt'
trainingSet, testingSet = data.getDataSets(training_path, testing_path)

####################################################################
learningRate = 0
inputNodes = len(testingSet[0].data)
hiddenNodes = 250
####################################################################


def softmax(x):
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

print "Letters Data Set ###############################"
for z in range(0,1):
    #hiddenNodes += 300
    learningRate += .05
    avg = 0
    iters = 1
    for y in range(0,iters):

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

            if False:
             print x.label, " : ", indices[np.argmax(guess)]
             for y in range(0,len(guess)):
                print guess[y]
             print exp[y] - guess[y]

            backPropagate(network, expected[x.label], softmax(guess), hiddenOut, inputOut)
            #raw_input("next")

            if count%7 == 0:
                #print "hello"
                for x in hiddenLayer: x.learnRate = x.learnRate * .65
                for x in outputLayer: x.learnRate = x.learnRate * .65



        count = 0
        for x in testingSet:
            guess, hiddenOut, inputOut = runDatum(network, x)
            print x.label, " : ", indices[np.argmax(softmax(guess))]
            if x.label == indices[np.argmax(guess)]:
                count += 1
        #print "accuracy: %f" %(float(count) / 23)
        avg += float(count)/ len(testingSet)
        #print indices[np.argmax(guess)]

        if False:
            #print x.label, " : ", indices[np.argmax(guess)]
            for y in range(0, len(guess)):
                print guess[y]
            print exp[y] - guess[y]
    print "learn rate: ", learningRate, " => ", avg / iters


###########################################################################

dataset = 'voting'



path = home + "voting_data.txt"
set = rd.readFromFile(path)
set = rd.parseToArrays(set)
sets = rd.splitIntoSets(set, 3)
train = sets[0] + sets[1]
test = sets[2]

newTrain = []
for x in train:
    label = x[0]
    del x[0]
    newTrain.append(data.datum(label,x))

newTest = []
for x in test:
    label = x[0]
    del x[0]
    newTest.append(data.datum(label,x))
data.normalizeVoting(newTrain)
data.normalizeVoting(newTest)

####################################################################
learningRate = 0
inputNodes = len(newTest[0].data)
hiddenNodes = 15
####################################################################
dataX = []
dataY = []

print "Voting Data Set  ############################### (EXTRA CREDIT)"
for z in range(0,200):
    #hiddenNodes += 1
    learningRate += .05
    avg = 0
    iters = 5
    for y in range(0,iters):

        network = []
        inputLayer = []
        hiddenLayer = []
        outputLayer = []
        for x in range(0, inputNodes): inputLayer.append(node.neuron(learningRate, True, False, False, 0))
        for x in range(0, hiddenNodes): hiddenLayer.append(node.neuron(learningRate, False, False, False, inputNodes+1))
        for x in range(0, 2): outputLayer.append(node.neuron(learningRate, False, True, False, hiddenNodes + 1))

        # add bias nodes to input and hidden layers
        inputLayer.append(node.neuron(learningRate, True, False, True, 0))
        hiddenLayer.append(node.neuron(learningRate, False, False, True, inputNodes + 1))

        network.append(inputLayer)
        network.append(hiddenLayer)
        network.append(outputLayer)

        expected = data.votingExpected()
        indices = data.votingIndex()

        count = 0
        for x in newTest:
            count += 1
            exp = expected[x.label]
            #print exp
            guess, hiddenOut, inputOut = runDatum(network, x)
            #print x.label, " : ", indices[np.argmax(guess)]
            #for y in range(0,len(guess)):
            #    print guess[y]
                #print exp[y] - guess[y]
            #print indices[x.label], softmax(guess)
            backPropagate(network, indices[x.label], softmax(guess), hiddenOut, inputOut)
            #raw_input("next")

            if count%20 == 0:
                #print "hello"
                for x in hiddenLayer: x.learnRate = x.learnRate * .65
                for x in outputLayer: x.learnRate = x.learnRate * .65



        count = 0
        for x in newTest:
            guess, hiddenOut, inputOut = runDatum(network, x)
            #print x.label, " : ", np.argmax(softmax(guess))
            if x.label == np.argmax(guess):
                count += 1
        avg += float(count)/ len(newTest)
    dataX.append(learningRate)
    dataY.append(avg)

    print z
    #print "hidden nodes: ", hiddenNodes, " => ", avg / iters
title='Effect of Learning Rate on Accuracy'
pl.plotDataScatter(title,dataX,dataY,"Learning Rate","Accuracy")




