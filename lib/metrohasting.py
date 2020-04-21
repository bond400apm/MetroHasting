import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

import numpy as np

import lib.pdfs as pdfs

#Using metrohasting algorithm
class MetroHasting_1D(object):
    '''
    This class contains 4 datas and two methods. 
    The starting point is defined by self.init_state (float)
    self.steps records total number of revolution steps (integer)
    self.sample records all the accepted sample(list of float)
    self.potential_state stores the temperate sample need to be
    further evaluated.
    '''

    def __init__(self):
        self.init_state = 4.0
        self.steps = 0
        self.sample = [self.init_state]
        self.potential_state = self.init_state
    
    #Generate the potential sample
    def gen_potential_state(self):
        x_t = self.sample[-1]
        _ = 0
        while _ < 1:
            randomnumber = np.random.normal(loc=x_t)
            if randomnumber>1.0:
                x = randomnumber
                _ = _ + 1
        self.potential_state = x

    #Decide whether or not accept the sample and move on
    def evolve(self,*data):
        p_c = np.exp(pdfs.likelihood(pdfs.gauspdf,self.potential_state,*data))
        p_t = np.exp(pdfs.likelihood(pdfs.gauspdf,self.sample[-1],*data))
        A = min(1,p_c/p_t)
        u = np.random.random_sample()
        if u <= A:
            self.sample.append(self.potential_state)
        else:
            self.sample.append(self.sample[-1])
        self.steps = self.steps + 1




