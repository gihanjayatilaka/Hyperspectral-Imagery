'''
Dataset Visualizer for HSI images

gihanjayatilaka@eng.pdn.ac.lk wrote this code. But if you find a bug you should notify harshana.w@eng.pdn.ac.lk
        because Harshana is the best debugger we know!

Ex:
python .\Dataset-Visualizer.py .\datasets\Botswana.mat Botswana .\datasets\Botswana_gt.mat Botswana_gt
python .\Dataset-Visualizer.py .\datasets\KSC.mat KSC .\datasets\KSC_gt.mat KSC_gt
python .\Dataset-Visualizer.py .\datasets\PaviaU.mat paviaU .\datasets\PaviaU_gt.mat paviaU_gt


'''


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
X=X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
Y=np.array(spio.loadmat(labelsFileName)[labelsMatName])
Y=Y.reshape((Y.shape[0]*Y.shape[1]))




classLabels=np.unique(Y)
noOfClasses=classLabels.shape[0]
noOfDimensions=X.shape[1]

numberToLabel=classLabels
labelToNumber=np.zeros(shape=(np.max(classLabels)+1),dtype=np.int32)

#print(classLabels,labelToNumber)
for i in range(classLabels.shape[0]):
    labelToNumber[classLabels[i]]=i


for i in range(Y.shape[0]):
    Y[i]=labelToNumber[Y[i]]


Y[:]=labelToNumber[Y[:]]

classMean=np.zeros(shape=(classLabels.shape[0],X.shape[1]),dtype=np.float32)
classVar=np.zeros(shape=(classLabels.shape[0],X.shape[1]),dtype=np.float32)
classCount=np.zeros(shape=(classLabels.shape[0]),dtype=np.uint32)
classSum=np.zeros(shape=(classLabels.shape[0],X.shape[1]),dtype=np.float32)


for i in range(X.shape[0]):
    classCount[Y[i]]+=1
    #print(Y[i].shape,classSum.shape,X[i,:].shape)
    classSum[Y[i],:]+=X[i,:]


for c in range(noOfClasses):
    for d in range(X.shape[1]):
        classMean[c,d]=np.divide(classSum[c,d],classCount[c])


for i in range(X.shape[0]):
    classVar[Y[i],:]+=np.power(X[i,:]-classMean[Y[i],:],2)

for c in range(noOfClasses):
    for d in range(X.shape[1]):
        classVar[c,d]=np.divide(classVar[c,d],classCount[c])
        if classVar[c,d] > np.mean(classVar[c,:])*3.0:
            classVar[c, d]=0
        #classVar[c, d]=np.min(classMean[c,d],classVar[c,d])

classStdDev=np.sqrt(classVar)


print("No of classes",noOfClasses,"Class count",classCount)
print("Class means",classMean)
print("Class std dev",classStdDev)



import matplotlib.pyplot as plt

plt.figure()

for c in range(noOfClasses):
    plt.subplot((noOfClasses/2)+(np.mod(noOfClasses,2)),2,c+1)

    plt.plot(np.arange(0,noOfDimensions),classMean[c,:],
            np.arange(0,noOfDimensions),classMean[c,:]+classStdDev[c,:],
            np.arange(0,noOfDimensions),classMean[c,:]-classStdDev[c,:])

plt.suptitle("Different classes, means and mean +- sigma")
plt.show()