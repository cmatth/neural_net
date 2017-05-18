import mnist
import numpy as np
import readData as rd
import plot as pl
import net as nt
import metic as metric

# Define Data Parameters #############################################
trainingSet, testingSet = mnist.getMNISTsets('/media/casey/Storage/MNIST')
expected = mnist.expectedMNISTOutputs()
# Define Net Parameters ##############################################
learningRate = .075
inputNodes = len(testingSet[0].data)
hiddenNodes = 784
outputNodes = 10
numEpochs = 600
batchSize = 100
numExperiments = 1
######################################################################

xData = []
yData = []
lYData = []
lXData = []
print "MNIST ###############################"
avgAcc = 0
avgC = 0
for y in range(0, numExperiments):
    net = nt.NeuralNet(inputNodes, hiddenNodes, 10, learningRate)
    #indices = data.labelIndex()
    #correct = data.correctIndex()

    epochs = 1
    numShown = 0
    while epochs <= numEpochs:
        # shuffle(trainingSet)
        Fout = []
        for x in trainingSet:
            numShown += 1
            print 'Example: %d / %d' %(numShown,len(trainingSet))
            exp = expected[x.label]
            net.forwardPropagate(x)
            # Decaying learning rate.
            # if numShown % len(testingSet) / 7 == 0:
            if numShown % batchSize == 0:
                net.backPropagate(expected[x.label], True)
                # net.learnRate = net.learnRate * .65
                # Get Results
                count = 0
                progress = 0
                for x in testingSet:
                    net.forwardPropagate(x)
                    progress += 1
                    print ('Test: %d%% : %d / %d' %(progress / len(testingSet),progress,len(testingSet)))
                    if x.label == np.argmax(net.finalOut): count += 1
                # TESTING ###################################################################
                Fout.append(net.finalOut[x.label])
                acc = float(count) / len(testingSet)
                print 'Epochs: ', epochs, '  Examples Shown: ', numShown, '  Accuracy: ', acc
                xData.append(epochs)
                yData.append(acc)
                #############################################################################
                if epochs >= 3:
                    avgAcc += acc
                    avgC += 1
            else:
                net.backPropagate(expected[x.label], False)
        lYData.append(metric.logloss(Fout))
        lXData.append(epochs)
        print 'Epoch: ', epochs, '  Log Loss: ', metric.logloss(Fout)
        epochs += 1
        if epochs > numEpochs: break
# print "Average Accuracy: ", " => ", avgAcc / avgC

pl.plotDataScatter('Epochs on Accuracy', xData,yData,'Epochs','Accuracy')
pl.plotDataScatter('Epoch on Log Loss',lXData,lYData,'Epochs','Log Loss')

# Save Network Weights
save = raw_input('Save this network? (\'Y\' to save)')
if save == 'Y':
    import writeData as wd

    wd.saveWeights(net.hiddenLayer, 'letter_hidden_weights')
    wd.saveWeights(net.outputLayer, 'letter_output_weights')
    print 'Network has been saved.'
else:
    pass