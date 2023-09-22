import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *

%matplotlib inline
%load_ext autoreload
%autoreload 2


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]



train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig[0],-1).T

def sigmoid(z):
    # z - A scalar or numpy array of any size.
    s = 1 / (1 + np.exp(z))
    
    return s

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0 # Because the default data type for numpy arrays is float
    
    return w,b

def propagation(w,b,X,Y):
    #w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    #b -- bias, a scalar
    #X -- data of size (num_px * num_px * 3, number of examples)
    #Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    m = X.shape[1]
    
    # forward propagation (from x to cost)
    z = np.dot(w.T,X) + b
    A = 1 / (1 + np.exp(-z)) 
    cost = - (1 / m) * (np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T))
    # backward propagation (to find grad)
    dz = A - Y
    dw = ( 1 / m ) * np.dot(X, dz.T)
    db = ( 1 / m ) * np.sum(dz)
    
    cost = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

def optimize(w, b, X, Y, num_iterations = 100, learning_rate = 0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagation(w,b,X,Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
                
    params = {"w":w, "b":b}
    grads = {"dw":dw , "db":db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    A = sigmoid(np.dot(w.T,X)+b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    
    if print_cost:
        print("train accuracy : {} %".format(100-np.mean(np.abs(Y_prediction_train - Y_train))*100))
        print("train accuracy : {} %".format(100-np.mean(np.abs(Y_prediction_test - Y_test))*100))
        
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_Prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    
    return d

