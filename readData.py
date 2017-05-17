import random
import copy
import os

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


def homefolder():
    l_top  = '/home/kmoney/Documents/neural_net/'
    d_top = '/home/casey/PycharmProjects/neural_net/'

    if   os.path.isdir(d_top): return d_top
    elif os.path.isdir(l_top): return l_top
    else:
        os.system('clear')
        print '################################################\n' \
              'There is no valid home folder defined. Enter a \n' \
              'valid system path that points to the directory \n' \
              'in which program data is stored (Q to Quit).\n' \
              '################################################'
        while(True):
            path = raw_input('Path: ')
            if   path == 'Q':         return False
            elif os.path.isdir(path): return path
            else: print 'Invalid Path.'

def readIDX(path):
    import idx2numpy
    ndarr = idx2numpy.convert_from_file(path)
    f_read = open(path, 'rb')
    ndarr = idx2numpy.convert_from_file(f_read)
    s = f_read.read()
    #ndarr = idx2numpy.convert_from_string(s)
    return ndarr
