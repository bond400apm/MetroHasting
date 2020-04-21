import numpy as np
from matplotlib import pyplot as plt

import lib.pdfs as pdfs
import lib.metrohasting as met

if __name__ == "__main__":

    #Take 100 random sample from standard normal distribution
    sample_num = 100
    post  = []
    x = np.random.normal(size=sample_num)

    '''
    Scan through possible parameter space 
    From a = 1.1 to a = 10, a is the esitmator of e
    For each value of a, calculate the posterior using Bayesian equation
    '''

    for a in np.linspace(1.1,10,num=200):
        post.append(pdfs.posterior(a,pdfs.likelihood,pdfs.gauspdf,*x))
    
    #Make the plot of posterior
    plot1 = plt.figure(1)
    plt.plot(np.linspace(1.1,10,num=200),post)
    plt.ylabel("Posterior probability density")
    
    #metro is an instance of MetroHasting Class, more information can be found in lib.metrohasting.py
    metro = met.MetroHasting_1D()
    for t in range(10000-1):
        metro.gen_potential_state()
        metro.evolve(*x)

    #Plot the resulting chain and histogram
    plot2 = plt.figure(2)
    plt.plot(range(10000),metro.sample)
    plt.ylabel("Parameter Chain")

    plot3 = plt.figure(3)
    plt.hist(metro.sample,bins=100)
    plt.ylabel("Count")
    plt.show()


            

