import random
import copy

def readFromFile(path):
    with open(path) as f:
        content = f.readlines()

    content = [x.strip('\n') for x in content]

    return content

# splits a single data set into n equal subsets randomly
def splitIntoSets(dataset, n):
    size = len(dataset) / n
    random.shuffle(dataset)
    collection = []
    index = 0
    for x in range(0,n):
        s = []
        for y in range (index, index+size):
            s.append(copy.copy(dataset[y]))
            index += 1
        collection.append(copy.deepcopy(s))
    return collection

def parseToArrays(dataset):
    newset = []
    for x in dataset:
        newset.append(x.split(','))
    return newset
