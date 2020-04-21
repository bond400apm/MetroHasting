import numpy as np
import math

# This defines a Gaussian pdf with mean value "a" and standard deviation 1
def gauspdf(x,a):
    '''
    input:x function variable(float)
    input:a Gaussian mean(float)
    output: The corrsponding value of x for the given Guassian mean(float)
    '''
    N = np.sqrt(np.log(a)/(2*np.pi))
    P = N*math.pow(a,-1*x*x/2)

    return P
#This estimate the likelihhod of a parameter value for a given function and data set 
def likelihood(funcs,parameter,*data):
    '''
    input:funcs  The single sample probability function(float)
    input:parameter  The parameter need to be estimated(float)
    input:*data  The given data set to estimate the likelihood(list or any number of float)
    output: likelihood(float)
    '''

    L = 1
    for x in data[1:]:
        L = L+np.log(funcs(x,parameter))
    return L
    
    #This function transfers a likelihood to a Bayesian posterior
def posterior(a,like,funcs,prior=1.0,normalization=1.0,*data):
    P = np.exp(like(funcs,a,*data))*prior/normalization
    return P
