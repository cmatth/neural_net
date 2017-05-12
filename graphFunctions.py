import plot as pl
from math import exp
from math import log

def bipolarSigmoid(x,param):
    return (1.0 - exp(-x * param)) / (1.0 + exp(-x * param))
def logbase2(x,param):
    return (log(x-param))


# Graphing Logarithmic
range = 5 # graph on x = [-range,range]
x = []
y = []
z= 0.1
param = 0
while z <= range:
    x.append(z)
    y.append(logbase2(z,param))
    z += .01
pl.plotDataScatter("Exponential",x,y,'X','Y')

