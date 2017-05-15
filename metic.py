from math import log

def logloss(preds):
    sum = 0
    for x in preds:
        sum += log(1-x)
    return (-1/len(preds)) * sum