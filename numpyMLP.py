from pandas import DataFrame
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import numpy as np
import os
import sys

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(77)

FTRAIN = '/home/soren/Desktop/kaggle/facialkeypoints/training.csv'
FTEST = '/home/soren/Desktop/kaggle/facialkeypoints/test.csv'
FLOOKUP = '/home/soren/Desktop/kaggle/facialkeypoints/IdLookupTable.csv'



def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None
    
    return X, y
X,y=load()
lr=1e-7
import matplotlib.pyplot as plt


def cheeky_fprop(X,W1,W2):
    """forward prop a neural net, just practicing """
    a = np.dot(X,W1)
    #print 'a shape: {}'.format(a.shape)
    z = np.maximum(a,0)

    h0 = np.dot(z,W2)
    return h0,z

def cheeky_bprop(X,W1,W2,y,h0,z):#WIT DA LADS 
    """backprop 2 layer net """
    dEdh0 = y - h0 #derivative of squared error wrt output is just error
    dW2 = np.dot(z.T,dEdh0)# ()
    dEdz = np.dot(W2,dEdh0.T) #not sure why this is true..?
    dzda = z > 0
    dEda = dEdz * dzda.T
    dW1 = np.dot(dEda,X)
    return dW1,dW2
    
def cheeky_nn_train(X,y,lr,n_iter=3,W1=None,W2=None,nHidden=100,split=300,lr_decay=1.0):
    """ Train those cheeky 2 layer nets"""
    plt.axis([0, n_iter/10, 0, 0.4])
    plt.ion()
    X,y = shuffle(X,y,random_state=95)
    Xv = X[0:split,:]
    yv = y[0:split,:]
    Xt = X[split:,:]
    yt = y[split:,:]
    if W1 is None and W2 is None:
        W1 = np.random.normal(scale = 1e-2, size = (X.shape[1],nHidden))
        W2 = np.random.normal(scale = 1e-2, size = (nHidden,y.shape[1]))
        print 'randomly intialised weight matrix'
    print('beginning training')
    i=0
    while i < n_iter:
        h0,z = cheeky_fprop(Xt,W1,W2)
        #print 'h0 shape: {} | z shape: {}'.format(h0.shape,z.shape)
        dW1,dW2 = cheeky_bprop(Xt,W1,W2,yt,h0,z)
        val_pred,_ = cheeky_fprop(Xv,W1,W2)
        val_error = yv - val_pred
        error=yt-h0
        #print val_error.shape
        #print 'train error shape: ' + str(error.shape)
        RMSE = (np.mean(np.square(error)))
        RMSEv = (np.mean(np.square(val_error)))
        lr *= lr_decay
        print( "Iteration: {} | train MSE: {!r} | Val MSE {!r} ".format(i,RMSE,RMSEv) )
        #Wwupdates = lr*np.dot(Xt.T,error)
        dW1=dW1.T
        W1 += lr * dW1
        W2 += lr * dW2
        if i % 10 == 0:
            plt.scatter(i,RMSEv)
            plt.scatter(i,RMSE,color='r')
        #plt.scatter(i,RMSE)
        plt.pause(0.001)
        i+=1
    return W1,W2
cheeky_nn_train(X,y,lr=1e-7,n_iter=1000)
