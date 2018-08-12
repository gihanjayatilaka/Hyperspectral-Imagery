import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy as sp
import scipy.io as spio

np.random.seed(7)


dataFileName=sys.argv[1]
dataMatName=sys.argv[2]
labelsFileName=sys.argv[3]
labelsMatName=sys.argv[4]

X=np.array(spio.loadmat(dataFileName)[dataMatName])
X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
Y=np.array(spio.loadmat(labelsFileName)[labelsMatName])
Y.reshape((Y.shape[0]*Y.shape[1]))


CLASSES=np.unique(Y)

classMean=np.ndarray((CLASSES.shape[0],X.shape[2]))
classVar=np.ndarray(((CLASSES.shape[0],X.shape[2])))





