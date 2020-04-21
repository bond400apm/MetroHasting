import numpy as np
import math

def gauspdf(x,a):
    N = np.sqrt(np.log(a)/(2*np.pi))
    P = N*math.pow(a,-1*x*x/2)
    if N < 0:
        print(N)
    return P

def likelihood(funcs,parameter,*data):
    L = 1
    for x in data[1:]:
        L = L+np.log(funcs(x,parameter))
    return L

def posterior(a,like,funcs,prior=1.0,normalization=1.0,*data):
    P = np.exp(like(funcs,a,*data))*prior/normalization
    return P
