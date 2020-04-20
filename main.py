import numpy as np
from matplotlib import pyplot as plt

import lib.pdfs as pdfs
import lib.sampling as sam

if __name__ == "__main__":
    sample_num = 100
    post  = []
    x = np.random.normal(size=sample_num)
    for a in np.linspace(1,10,num=200):
        post.append(pdfs.likelihood(pdfs.gauspdf,a,*x))
    
    plt.plot(np.linspace(1,10,num=200),post)
    plt.ylabel("Posterior probability density")
    plt.show()

            

