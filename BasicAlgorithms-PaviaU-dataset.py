'''
USAGE:
python BasicAlgorithms-PaviaU-dataset.py TRAINING_SAMPLE_SIZE EPOCHS ALGORITHM
python BasicAlgorithms-PaviaU-dataset.py 500 100 cnn
python BasicAlgorithms-PaviaU-dataset.py 250 0 svm


Note: EPOCHS are useful only for neural networks.
Neural network algorithms implemented: cnn,dnn
Classical algorithms implemented: dectree,svm,ranfor,lle
'''


from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D,Flatten,Reshape
import scipy.io
import numpy as np
from keras.utils.vis_utils import plot_model
import numpy
from keras.models import Sequential
import scipy.io
import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import random
import numpy,keras
import matplotlib.pyplot as plt
numpy.random.seed(7)
import sys




TRAINING_SAMPLE=int(sys.argv[1])
np.random.seed(7)

def nn(DIMENSIONS):
    model = Sequential()

    if DIMENSIONS==103:
        model.add(Dense(103, input_dim=103, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='softmax'))
    return model

def cnn():
    model = Sequential()
    model.add(Conv1D(filters=20, kernel_size=11, input_shape = (103, 1), activation='tanh'))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(9, activation='softmax'))

    return model

paviaU=np.array(scipy.io.loadmat('datasets/paviaU.mat')['paviaU'])
paviaUgt=np.array(scipy.io.loadmat('datasets/paviaU_gt.mat')['paviaU_gt'])-1

paviaU=paviaU.reshape([paviaU.shape[0]*paviaU.shape[1],paviaU.shape[2]])
paviaUgt=paviaUgt.reshape((paviaUgt.shape[0]*paviaUgt.shape[1],1))

count=0
for x in range(paviaU.shape[0]):
    if paviaUgt[x][0]>=0 and paviaUgt[x][0]<255:
        count=count+1

X=np.ndarray((count,paviaU.shape[1]))
Y=np.ndarray((count,1),dtype=np.int32)
count=0
for x in range(paviaU.shape[0]):
    if paviaUgt[x][0]>=0 and paviaUgt[x][0]<255:
        X[count,:]=paviaU[x,:]
        Y[count,:]=np.int(np.round(paviaUgt[x,:]))

        count=count+1
print('count=',count)

CLASSES=(np.unique(Y)).shape[0]

countClass=np.ndarray((CLASSES))
for x in range(CLASSES):
    countClass[x]=0
for x in range(Y.shape[0]):
    countClass[Y[x,0]]=countClass[Y[x,0]]+1

print('No of classes', countClass)






X = X / np.max(X, axis=0)


Xtrain=np.ndarray((TRAINING_SAMPLE*CLASSES,X.shape[1]))
Ytrain=np.ndarray((TRAINING_SAMPLE*CLASSES,1),dtype=np.int32)
count=0
for x in range(CLASSES):
    countClass[x]=0
while count<TRAINING_SAMPLE*CLASSES:
    r=np.random.randint(0,X.shape[0])
    if countClass[Y[r]] != TRAINING_SAMPLE:
        Xtrain[count]=X[r]
        Ytrain[count]=Y[r]
        countClass[Y[r]]=countClass[Y[r]]+1
        count=count+1


if sys.argv[3]=='cnn' or sys.argv[3]=='dnn' or sys.argv[3]=='ldann':
    Y = keras.utils.to_categorical(Y, np.max(Y) + 1)
    Ytrain = keras.utils.to_categorical(Ytrain, np.max(Ytrain) + 1)

if sys.argv[3]=='cnn':
    XX = X
    XXtrain = Xtrain
    X=np.ndarray((X.shape[0], X.shape[1], 1))
    Xtrain=np.ndarray((Xtrain.shape[0], Xtrain.shape[1], 1))
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i,j,0]=XX[i,j]

    for i in range(len(Xtrain)):
        for j in range(len(Xtrain[0])):
            Xtrain[i,j,0]=XXtrain[i,j]



xx=input('Start training?')

if sys.argv[3]=='cnn' or sys.argv[3]=='dnn':
    if sys.argv[3]=='cnn':
        model = cnn()
    if sys.argv[3]=='dnn':
        model = nn(103)


    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    print(model.summary())

    # Fit the model
    history = model.fit(Xtrain, Ytrain, epochs=int(sys.argv[2]), batch_size=32, shuffle=True, validation_split=0)

    # evaluate the model
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Loss Curves
    '''plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    plt.show()'''

else:
    if sys.argv[3]=='dectree':
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()

        clf.fit(Xtrain, Ytrain)
        pre=clf.predict(X)
    elif sys.argv[3]=='svm':
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(Xtrain, Ytrain.ravel())
        pre = clf.predict(X)
    elif sys.argv[3]=='ranfor':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=50, random_state=0)
        clf.fit(Xtrain, Ytrain.ravel())
        pre=clf.predict(X)

    elif sys.argv[3]=='lle':
        from sklearn.manifold import LocallyLinearEmbedding
        lle = LocallyLinearEmbedding(n_neighbors=int(round(TRAINING_SAMPLE/5)),n_components=50)
        lle.fit(Xtrain,Ytrain)
        Xtrain=lle.transform(Xtrain)
        X=lle.transform(X)

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=50, random_state=0)
        clf.fit(Xtrain, Ytrain.ravel())
        pre = clf.predict(X)



    correct=0
    wrong=0
    for x in range(len(pre)):
        if pre[x]==Y[x]:
            correct=correct+1
        else:
            wrong=wrong+1

    print('correct',correct,'wrong',wrong,'percentage',((100*correct)/(correct+wrong)))
