import numpy as np

def sigmoid(z):
    return (1/(1+np.exp(-z)))


def calc_gradient(theta,X,y):
    m=y.size

    return (X.T @ (sigmoid(X@theta)- y))/m

def gradient_descent(X,y,alpha=0.1,no_iterations=100,epsilon=1e-7):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    theta=np.zeros(X_b.shape[1])

    for i in range (no_iterations):
        grad=calc_gradient(theta,X_b,y)
        theta-=alpha*grad

        if np.linalg.norm(grad)<epsilon:
            break

    return theta
    
def predict_prob(X,theta):
    X_b=np.c_[np.ones((X.shape[0],1)),X]
    return sigmoid(X_b@theta)

def predict(X,theta,threshold=0.5):
    return predict_prob(X,theta)>=threshold

