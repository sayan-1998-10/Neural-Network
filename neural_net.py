import numpy as np
import scipy.io as sio
import pandas as pd
from skimage.io import imshow
from skimage.color import rgb2yiq
from PIL import Image
epsilon = 1e-11 #to avoid NaN , inf

#def predict(theta1,theta2,)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost_Func(X,Y,theta1,theta2):


    ones = np.ones((5000,1))
    a1 = np.hstack((ones,X)) #first input layer and adding bias unit
    a2 = sigmoid(a1 @ theta1.T) #hidden layer
    a2 = np.hstack((ones,a2))
    a3 = sigmoid(a2 @ theta2.T)

    sum = np.multiply(Y , np.log(a3+epsilon))+ np.multiply((1-Y), np.log(1-a3+epsilon))
    temp_J = np.sum(sum)

    J = np.sum(temp_J)/(-len(Y))    #cost

    return J




def predict(X,model):
    m = X.shape[0]
    theta1_new,theta2_new = model
    ones = np.ones((m,1))
    a1 = np.hstack((ones,X)) #first input layer
    z2 = a1 @ theta1_new.T
    a2 = sigmoid(z2)    #hidden layer
    a2 = np.hstack((ones,a2))
    a3 = sigmoid(a2 @ theta2_new.T)
    print(np.round(a3[0]))
    return np.argmax(a3, axis = 1)


def grad_sigmoid(z):
   
    return sigmoid(z)*(1-sigmoid(z))


#implementing backpropagation
def error(X,Y,theta1,theta2,reg):
    
    #forward propagation-------------------
    m = X.shape[0]
    ones = np.ones((m,1))
    a1 = np.hstack((ones,X)) #first input layer
    z2 = a1 @ theta1.T
    a2 = sigmoid(z2)    #hidden layer
    a2 = np.hstack((ones,a2)) 
    a3 = sigmoid(a2 @ theta2.T)#    zz2 = np.hstack((ones,z2))
    #error term for last layer
    last_delta = a3 - Y
    #error term for hidden layer---
    second_delta = ((theta2[:,1:].T @ last_delta.T).T )* grad_sigmoid(z2)
    #---------------------------
    #calculating the partial derivatives
    D1 = np.zeros_like(theta1)
    D2 = np.zeros_like(theta2)

    D1=second_delta.T @ a1
    D2=last_delta.T   @ a2

    #regularised
    grad1  = (1/m)*(D1  + reg*theta1)
    grad2  = (1/m)*(D2  + reg*theta2)
    
    return (grad1,grad2)

def gradient_descent(X,Y,alpha=0.8,iterations=100):
    theta1 = randInitializeWeights(25,401) #25*401
    theta2 = randInitializeWeights(10,26)#10*26
#running iterations over theta values
    for i in range(iterations):
        cost= cost_Func(X,Y,theta1,theta2)
        grad1,grad2 = error(X,Y,theta1,theta2,reg=0)
        theta1 = theta1 - alpha*grad1
        theta2 = theta2 - alpha*grad2
        print("Iteration no.:{0},learning_rate:{1},cost: {2}".format((i+1),alpha,cost))
    return cost,(theta1,theta2)

def train(X,Y,alpha=0.01,iterations=100):
    cost,model = gradient_descent(X,Y,alpha,iterations)
    return model

def randInitializeWeights(L_out,L_in):

    randWeights = np.random.uniform(low=-.12,high=.12,
                                    size=(L_out,L_in))
    return randWeights

def main():
    data = sio.loadmat("C:/Users/europ/Desktop/machine-learning-ex4/ex4/ex4data1.mat")
#    weights = sio.loadmat("C:/Users/europ/Desktop/machine-learning-ex4/ex4/ex4weights.mat")
    X = data['X']
    Y = data['y']
    Y = pd.get_dummies(Y.flatten())
    Y=Y.values

    model = train(X,Y,alpha=2.0,iterations=300)

    pred=predict(X,model)
    #calculating mean
    print("Training Set Accuracy:",sum(pred[:,np.newaxis]==np.argmax(Y,axis=1)[:,np.newaxis])/5000*100,"%")

if __name__=='__main__':
    main()
