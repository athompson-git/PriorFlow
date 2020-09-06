# example.py
# An example program to showcase the prior-flow technique and using copulas in Bayesian inference.
# USAGE: python example.py

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm, expon

#from copula import EmpricalCopula

import pymultinest
from pymultinest import Analyzer


##################### EXPERIMENT A (Ex. A)

# Make some pseudodata.
peak1Norm = norm(scale=2, loc=80)
peak2Norm = norm(scale=10, loc=30)
bgNorm = expon(scale=30)

rawDataA = np.append(peak1Norm.rvs(400), peak2Norm.rvs(25000)).clip(min=0.0)
rawDataA = np.append(rawDataA, bgNorm.rvs(50000).clip(min=0.0))

bin_edges = np.linspace(5, 120, 100)
bins = (bin_edges[1:] + bin_edges[:-1])/2
binnedDataA = np.histogram(rawDataA, bins=bin_edges)


# Define an events generator with the physics model for the spectrum (with 2 N.P. params) and additional nuisance params.
def GausBkg(x, x0, s, n):
    return n * (1/np.sqrt(2*np.pi)/s) * np.exp(-(x - x0)**2 / (2*s))

def FallingBkg(x, b, n):
    return n * (1/b)*np.exp(-x/b)

def EventsGenA(cube):
    np_loc = cube[0] # new physics location (~mass)
    np_size = cube[1] # new physics normalization (~coupling)
    
    fbkg_nuis = cube[2]  # background systematics nuisance parameter
    gausbkg_nuis = cube[3]  # background systematics nuisance parameter
    
    background = GausBkg(bins, 30, 95, 260000*(1+fbkg_nuis)) + FallingBkg(bins, 30, 60000*(1+gausbkg_nuis))
    signal = GausBkg(bins, np_loc, 2, np_size)
    return signal + background

# Plot the pseudodata for Ex. A and the background and signal models.
plt.plot(bins, EventsGenA([80, 0]), color="b", label="Background Model")
plt.plot(bins, EventsGenA([80, 400]), color="r", label="Background + Signal Model")
plt.errorbar(bins, binnedDataA[0], yerr=np.sqrt(binnedDataA[0]), color="k", label="Pseudodata Ex. A")
plt.xlabel(r"Energy $E$ [keV]")
plt.ylabel(r"Counts")
plt.legend()
plt.show()


# Define flat priors over NP params and gaussian priors over background systematics.
falling_bkg_syst = norm(scale=0.1)
gaus_bkg_syst = norm(scale=0.05)

def PriorsA(cube, N, D):
    cube[0] = 120*cube[0]
    cube[1] = 1000*cube[1]
    cube[2] = falling_bkg_syst.ppf(cube[2])
    cube[3] = gaus_bkg_syst.ppf(cube[3])


# Define a log-poisson likelihood to test against the data given the background+signal model.
def LogPoisson(cube, N, D):
    binnedSignalBkg = EventsGenA(cube)
    ll = binnedDataA * log(binnedSignalBkg) - binnedSignalBkg - gammaln(binnedDataA+1)
    return np.sum(ll)


# Run the fit.
pymultinest.run(LogPoisson, PriorsA, 4, outputfiles_basename="multinest/example/exampleOut", resume=False,
                verbose=True, n_live_points=1000, evidence_tolerance=0.5, sampling_efficiency=0.8)




##################### EXPERIMENT 2 (Ex. 2)

# We save the posterior data from the first fit on experiment 1 and define copulas over the new physics parameters.

# Define the events generator with the physics model for Ex. 2 (with 3 N.P. params).


# Define priors using empirical copulas over the posterior data from Ex. 1.


# Run the fit.



# Plot the 1-D and 2-D marginal posteriors for Ex. 2 comparing with Ex. 1.