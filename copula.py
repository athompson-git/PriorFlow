import numpy as np

# Using copulas in R is very simple. We can do it in python too by importing the rpy2 module.
import rpy2

# We can talk to the R language by importing robjects
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, ListVector

# Import all the base packages from R. If you have R installed, these should already be there.
from rpy2.robjects import r
base = importr('base')
utils = importr('utils')

# Import the copula package. This needs to be installed in R first.
rcopula = importr('copula')


# Define a class to construct an Empirical copula given a set of sample data (from MultiNest).
class EmpricalCopula:
    def __init__(self, datastr, i, j):
        robjects.r('data = read.table(file = "{0}", header=F)'.format(datastr))
        robjects.r('z = pobs(as.matrix(cbind(data[,{0}],data[,{1}])))'.format(i, j))
    
    def ddv(self, u, v2):
            robjects.r('u = matrix(c({0}, {1}), 1, 2)'.format(u, v2))
            return np.asarray(robjects.r('dCn(u, U = z, j.ind = 1)'))[0]
    
    def simulate(self, u, v):
        v_list = np.linspace(0,1,25)
        ddv_list = [self.ddv(u, _v) for _v in v_list]
        return np.interp(v, ddv_list, v_list)
        