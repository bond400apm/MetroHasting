import numpy as np
from matplotlib import pyplot as plt

import lib.pdfs as pdfs
import lib.metrohasting as met

if __name__ == "__main__":
    sample_num = 100
    post  = []
    x = np.random.normal(size=sample_num)

    for a in np.linspace(1.1,10,num=200):
        post.append(pdfs.posterior(a,pdfs.likelihood,pdfs.gauspdf,*x))
    
    plot1 = plt.figure(1)
    plt.plot(np.linspace(1.1,10,num=200),post)
    plt.ylabel("Posterior probability density")
    

    metro = met.MetroHasting_1D()
    for t in range(10000-1):
        metro.gen_potential_state()
        metro.evolve(*x)

    plot2 = plt.figure(2)
    plt.plot(range(10000),metro.sample)
    plt.ylabel("Parameter Chain")

    plot3 = plt.figure(3)
    plt.hist(metro.sample,bins=100)
    plt.ylabel("Count")
    plt.show()


            

