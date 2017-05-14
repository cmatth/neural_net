import data
#import node
import numpy as np
import readData as rd
import plot as pl
import net as nt

# Define Runtime Parameters ##########################################
LetterSet = True
VotingSet  = False
# Define Data Parameters #############################################
home = rd.homefolder()
training_path = home + 'training.txt'
testing_path =  home + 'testing.txt'
trainingSet, testingSet = data.getDataSets(training_path, testing_path)
expected = data.classOutputs()
# Define Net Parameters ##############################################
learningRate = .15
inputNodes = len(testingSet[0].data)
hiddenNodes = 250
######################################################################

if LetterSet:
    xData = []
    yData = []
    print "Letters Data Set ###############################"
    for z in range(0, 1):
        avg = 0
        iters = 2
        for y in range(0, iters):

            net = nt.NeuralNet(63,hiddenNodes,7,learningRate)
            expected = data.classOutputs()
            indices = data.labelIndex()

            count = 0
            for x in trainingSet:
                count += 1
                exp = expected[x.label]
                net.forwardPropagate(x)
                net.backPropagate(expected[x.label])

                # Decaying learning rate.
                if count % 7 == 0:
                    net.learnRate = net.learnRate * .65

            count = 0
            for x in testingSet:
                net.forwardPropagate(x)
                if x.label == indices[np.argmax(net.finalOut)]: count += 1
                print x.label, indices[np.argmax(net.finalOut)]
                print count
            print count, '/', len(testingSet)
            acc = float(count) / len(testingSet)
            print acc
            avg += acc
            print avg
            xData.append(hiddenNodes)
            yData.append(avg)

        print "Accuracy: ", " => ", avg / iters
    #pl.plotDataScatter('Hidden Nodes on Accuracy', xData,yData,'Hidden Nodes','Accuracy')

    ###########################################################################

if VotingSet:

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
        newTrain.append(data.datum(label, x))

    newTest = []
    for x in test:
        label = x[0]
        del x[0]
        newTest.append(data.datum(label, x))
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
    for z in range(0, 200):
        # hiddenNodes += 1
        learningRate += .05
        avg = 0
        iters = 5
        for y in range(0, iters):

            network = []
            inputLayer = []
            hiddenLayer = []
            outputLayer = []
            for x in range(0, inputNodes): inputLayer.append(node.neuron(learningRate, True, False, False, 0))
            for x in range(0, hiddenNodes): hiddenLayer.append(
                node.neuron(learningRate, False, False, False, inputNodes + 1))
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
                # print exp
                guess, hiddenOut, inputOut = net.runDatum(network, x)
                # print x.label, " : ", indices[np.argmax(guess)]
                # for y in range(0,len(guess)):
                #    print guess[y]
                # print exp[y] - guess[y]
                # print indices[x.label], softmax(guess)
                net.backPropagate(network, indices[x.label], net.softmax(guess), hiddenOut, inputOut)
                # raw_input("next")

                if count % 20 == 0:
                    # print "hello"
                    for x in hiddenLayer: x.learnRate = x.learnRate * .65
                    for x in outputLayer: x.learnRate = x.learnRate * .65

            count = 0
            for x in newTest:
                guess, hiddenOut, inputOut = net.runDatum(network, x)
                # print x.label, " : ", np.argmax(softmax(guess))
                if x.label == np.argmax(guess):
                    count += 1
            avg += float(count) / len(newTest)
        dataX.append(learningRate)
        dataY.append(avg)

        print z
        # print "hidden nodes: ", hiddenNodes, " => ", avg / iters
    title='Effect of Learning Rate on Accuracy'
    pl.plotDataScatter(title,dataX,dataY,"Learning Rate","Accuracy")