import numpy as np
import node

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





