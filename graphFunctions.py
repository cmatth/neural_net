import plot as pl
from math import exp
from math import log

def bipolarSigmoid(x,param):
    return (1.0 - exp(-x * param)) / (1.0 + exp(-x * param))
def logbase2(x,param):
    return (log(x-param))


# Graphing Logarithmic
range = 15 # graph on x = [-range,range]
x = []
y = []
z= -range
param = .5
while z <= range:
    x.append(z)
    y.append(bipolarSigmoid(z,param))
    z += .01
pl.plotDataScatter("Exponential",x,y,'X','Y')

