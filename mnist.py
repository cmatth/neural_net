
def outputTrainSet():
    import readData
    from termcolor import colored
    import sys
    from os import system
    import numpy as np

    training_path = '/media/casey/Storage/MNIST/train-images.idx3-ubyte'
    trainingL_path = '/media/casey/Storage/MNIST/train-labels.idx1-ubyte'
    train_set = readData.readIDX(training_path)
    labels = readData.readIDX(trainingL_path)

    for ind in range(len(train_set)):
        count = 0
        print '############', labels[ind], '#############'
        for z in range(len(train_set[ind])):
            for y in range(len(train_set[ind][z])):
                if train_set[ind][z][y] == 0:
                    sys.stdout.write(colored('0', 'blue', attrs=['dark']))
                    sys.stdout.flush()
                    count += 1
                else:
                    sys.stdout.write(colored('#', 'red'))
                    sys.stdout.flush()
                    count += 1
            sys.stdout.write('\n')
            sys.stdout.flush()
        raw_input(str(count) + ' pixels')
        system('clear')

def getMNISTsets(path):
    import readData
    import numpy as np
    import data
    trainSet = readData.readIDX(path + '/train-images.idx3-ubyte')
    trainLabs= readData.readIDX(path + '/train-labels.idx1-ubyte')
    testSet  = readData.readIDX(path + '/t10k-images.idx3-ubyte')
    testLabs = readData.readIDX(path + '/t10k-labels.idx1-ubyte')
    #trainSet.flags.writeable = True

    newTrain = []
    for x in range(len(trainSet)):
        newTrain.append(data.datum(trainLabs[x],np.reshape(trainSet[x], (1,784))[0]))
    newTest = []
    for x in range(len(testSet)):
        newTrain.append(data.datum(testLabs[x],np.reshape(testSet[x],(1,784))[0]))

    return newTrain,newTest