import numpy as np



def Linear(x):
    return x

def Linear_Deriv(x: np):
    return 1



def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Sigmoid_Deriv(x):
    sigmoid = Sigmoid(x)
    return sigmoid * (1 - sigmoid)



def ReLU(x: np):
    return np.maximum(x, 0)

def ReLU_Deriv(x: np):
    return np.where(x >= 0, 1, 0)



def LeakyReLU(x: np, a: float = 0.1):
    return np.where(x >= 0, x, x * a)

def LeakyReLU_Deriv(x: np, a: float = 0.1):
    return np.where(x >= 0, 1, a)



def Tanh(x: np):
    double_e = np.exp(2 * x)
    return (double_e - 1) / (double_e + 1)

def Tanh_Deriv(x: np):
    return 1 - Tanh(x) ** 2


