# Package imports
import numpy as np
import copy
import matplotlib.pyplot as plt
from testCases_v2 import *
from public_tests import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

%matplotlib inline

%load_ext autoreload
%autoreload 2

X, Y = load_planar_dataset()

# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

#Train the logistic regression classifier
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

#Plot the decision boundary for logisitic regression
plot_decision_boundary(lambda x: clf.predict(x),X,Y)
plt.title("Logistic Regression")

#Print accuracy
LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' 
      % float((np.dot(Y,LR_predictions) + np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) 
      + '% ' + "(percentage of correctly labelled datapoints)")

#GRADED FUNCTION : layer_sizes
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)

t_X, t_Y = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(t_X, t_Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the input layer is: n_h = " + str(n_h))
print("The size of the input layer is: n_y = " + str(n_y))

layer_sizes_test(layer_sizes)

#GRADED FUNCTION : initialize_parameters
def initialize_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2
                  "b2": b2}
    return parameters

np.random.seed(2)
n_x, n_h, n_y = initialize_parameters_test_case()
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

initialize_parameters_test(initialize_parameters)


#GRADED FUNCTION : forward_propagation
def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}
    
    return A2, cache

def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = - np.sum(logprobs) / m
    
    cost = float(np.squeeze(cost))
    
    return cost

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    #backward propagation: calculate dW1,db1,dW2, db2. (vetorized implementation)
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2,A1.T) / m
    db2 = np.sum(dZ2,axis=1, keepdims = True) / m
    dZ1 = np.dot(W2.T,db2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X.T) / m
    db1 = np.sum(dZ1,axis=1, keepdims = True) / m 
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

def update_parameters(paramerts, grads, learning_rate = 1.2):
    W1 = copy.deepcopy(paramerts["W1"])
    b1 = paramerts["b1"]
    W2 = copy.deepcopy(paramerts["W2"])
    b2 = paramerts["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
    
    return paramerts

#GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
            
    return parameters

#GRADED FUNCTION: predict
def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    # 이 줄에서 A2 배열의 각 원소가 0.5보다 큰지 확인
    # 만약 원소가 0.5보다 크면 True를 반환하고, 그렇지 않으면 False를 반환
    # 여기서 True는 Python에서 1로, False는 0으로 취급
    predictions = (A2 > 0.5)
    return predictions