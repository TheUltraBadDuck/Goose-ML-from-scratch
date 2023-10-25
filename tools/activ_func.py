import numpy as np



def Linear(x):
    return x

def Linear_Deriv(x: np):
    return 1



def Sigmoid(x):
    return 1 / (1 + np.exp(-x))





def ReLU(x: np):
    return np.maximum(x, 0)

def ReLU_Deriv(x: np):
    return np.where(x >= 0, 1, 0)









