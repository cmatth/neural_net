import json

def saveNetwork(network,dir):
    import os
    params = {
        'numInputNeurons' : len(network.inputLayer),
        'numHiddenNeurons': len(network.hiddenLayer),
        'numOutputNeurons': len(network.outputLayer),
        'learningRate'    : network.learnRate,
        'hiddenWeights'   : '_hidden_weights',
        'outputWeights'   : '_output_weights'}

    if not os.path.exists(dir): os.makedirs(dir)
    with open(dir + '/params', 'w') as outfile: json.dump(params,outfile)
    saveWeights(network.hiddenLayer, dir + '/_hidden_weights')
    saveWeights(network.outputLayer, dir + '/_output_weights')

def saveWeights(layer,path):
    with open(path, 'w') as outfile:
        weights = []
        for x in layer:
            weights.append(x.weights)
        json.dump(weights,outfile)

def loadWeights(filename):
    with open(filename,'r') as infile:
        return json.load(infile)

def loadNetwork(filename):
    import net
    with open(filename + '/params') as infile:
        params = json.load(infile)
    network = net.NeuralNet(params['numInputNeurons'],
                            params['numHiddenNeurons'],
                            params['numOutputNeurons'],
                            params['learningRate'])
    hiddenWs = loadWeights(filename + '/_hidden_weights')
    outputWs = loadWeights(filename + '/_output_weights')
    for x in len(network.hiddenLayer):
        network.hiddenLayer[x].weights = hiddenWs[x]
    for x in len(network.outputLayer):
        network.outputLayer[x].weights = outputWs[x]
    return network







