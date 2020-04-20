import numpy as np
from matplotlib import pyplot as plt

import lib.pdfs as pdfs
import lib.sampling as sam

if "__name__" == "__main__":
    sample_num = 100
    post  = []
    for a in np.linspace(1,10):
        x = sam.random_sampling(-10,10,pdfs.gauspdf,a)
        post.append(pdfs.posterior(a,pdfs.likelihood,pdfs.gauspdf,x))

    plt.plot(np.linespace(1,10),post)
    plt.ylabel("Posterior probability density")
    plt.show()
            

