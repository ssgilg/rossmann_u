#here metrics
import numpy as np


def bias_s(something, dif_):
    return dif_.mean()


def biasp_s(real, dif_):
    return dif_.sum()/real.sum()


def mape(real, dif_):
    return (dif_/real).sum()/len(real)

def mae(something, dif_):
    return (dif_.abs()).mean()

def maep(real, dif_):
    return (dif_.abs()).sum()/real.sum()

# def rmse(real, dif_):
#     return np.sqrt((dif_**2).mean())

# def rmsep(real, dif_):
#      return np.sqrt((dif_**2).mean())/real.mean()

    
def rmse(real, y_):
    dif_ = real - y_
    return np.sqrt((dif_**2).mean())



def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat / y-1) ** 2))

                   
def rmsep(real, y_):
    dif_ = real - y_
    return np.sqrt((dif_**2).mean())/real.mean() 
    
    
def mse(nocare, dif_):
    return (dif_**2).mean()