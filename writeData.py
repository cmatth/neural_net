import json

def saveWeights(layer,filename):
    with open('weights/' + filename, 'w') as outfile:
        weights = []
        for x in layer:
            weights.append(x.weights)
        json.dump(weights,outfile)
def loadWeights(filename):
    with open('weights/' + filename,'r') as infile:
        return json.load(infile)

