import sys
import time
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as PathEffects
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pymultinest
import mpi4py

from MNStats import MNStats

from copula import EmpricalCopula

from pyCEvNS.events import*
from pyCEvNS.oscillation import*


# Set global vars.
n_z = 20
max_z = -0.025
kTon = 40
years = 10
days_per_year = 365
kg_per_kton = 1000000
exposure = years * days_per_year * kTon * kg_per_kton




# Read in the posterior distributions from the COHERENT fit
coherent_path = "multinest/coherent/"
coherent_out_str = coherent_path + "coherent.txt"
coherent_wgts_str = coherent_path + "coherentpost_equal_weight.dat"

# TODO (me): manually get marginals and deprecate MNStats?
coherent_posterior = np.genfromtxt(coherent_out_str)
cohStats = MNStats(coherent_posterior)
xen_uee, xen_uee_cdf = cohStats.GetMarginal(0)[0], np.cumsum(cohStats.GetMarginal(0)[1])
xen_umm, xen_umm_cdf = cohStats.GetMarginal(1)[0], np.cumsum(cohStats.GetMarginal(1)[1])
xen_dee, xen_dee_cdf = cohStats.GetMarginal(6)[0], np.cumsum(cohStats.GetMarginal(6)[1])
xen_dmm, xen_dmm_cdf = cohStats.GetMarginal(7)[0], np.cumsum(cohStats.GetMarginal(7)[1])

# Set up empirical copulas that will model the COHERENT posteriors as our priors on the up and down ee/mumu NSI.
emp_ud_ee = EmpricalCopula(coherent_wgts_str, 1, 7)
emp_ud_mm = EmpricalCopula(coherent_wgts_str, 2, 8)




# Define the events generator
det = Detector("dune")
zenith_arr = np.round(np.linspace(-0.975,max_z,n_z), decimals=3)
energy_arr = np.array([106.00, 119.00, 133.00, 150.00, 168.00, 188.00, 211.00, 237.00, 266.00, 299.00,
                       335.00, 376.00, 422.00, 473.00, 531.00, 596.00, 668.00, 750.00, 841.00, 944.00])

osc_params = OSCparameters()
default_flux = NeutrinoFlux()
gen = NeutrinoNucleonCCQE("mu", default_flux)

# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, expo, flux, osc_factory):
    obs = np.zeros((n_z,energy_arr.shape[0]-1))  # 18 energy bins, 20 zenith bins
    nsi = NSIparameters(0)
    nsi.epe = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2],
              'em': nsi_array[3], 'et': nsi_array[4], 'mt': nsi_array[5]}
    nsi.epu = {'ee': nsi_array[6], 'mm': nsi_array[7], 'tt': nsi_array[8],
              'em': nsi_array[9], 'et': nsi_array[10], 'mt': nsi_array[11]}
    nsi.epd = {'ee': nsi_array[12], 'mm': nsi_array[13], 'tt': nsi_array[14],
              'em': nsi_array[15], 'et': nsi_array[16], 'mt': nsi_array[17]}


  # Begin event loop.
    for i in range (0, zenith_arr.shape[0]):
        osc = osc_factory.get(oscillator_name='atmospheric', zenith=zenith_arr[i], nsi_parameter=nsi,
                              oscillation_parameter=osc_params)
        transformed_flux = osc.transform(flux[i])
        gen.flux = transformed_flux

        for j in range (0, energy_arr.shape[0]-1):
            obs[i][j] = 2*pi*gen.events(energy_arr[j], energy_arr[j-1], det, expo)
    return obs





# Define the prior.
def PriorFlow18DEmpirical(cube, D, N):
    d_ee = emp_ud_ee.simulate(cube[0], cube[1])
    d_mm = emp_ud_mm.simulate(cube[2], cube[3])

    cube[0] = np.interp(cube[6], xen_uee_cdf, xen_uee)
    cube[1] = np.interp(cube[7], xen_umm_cdf, xen_umm)
)
    cube[12] = np.interp(d_ee, xen_dee_cdf, xen_dee)
    cube[13] = np.interp(d_mm, xen_dmm_cdf, xen_dmm)

  



def main():
    # Set up factories.
    osc_factory = OscillatorFactory()
    flux_factory = NeutrinoFluxFactory()

    # Prepare flux.
    z_bins = np.round(np.linspace(-0.975, max_z, n_z), decimals=3)
    flux_list = []
    for z in range(0, z_bins.shape[0]):
      this_flux = flux_factory.get('atmospheric', zenith=z_bins[z])
      flux_list.append(this_flux)

    # Construct test data.
    sm_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sm_events = EventsGenerator(sm_params, exposure, flux_list, osc_factory)
    null = sm_events.flatten()
    width = np.sqrt(null/2) + 1

    def LogLikelihood(cube, D, N):
        signal = (EventsGenerator(cube, exposure, flux_list, osc_factory)).flatten()
        likelihood = -0.5 * np.log(2 * pi * width ** 2) - 0.5 * ((signal - null) / width) ** 2
        return np.sum(likelihood)
    
    # Define model parameters
    parameters = ["eps_e_ee", "eps_e_mumu", "eps_e_tautau", "eps_e_emu", "eps_e_etau", "eps_e_mutau",
                  "eps_ud_ee", "eps_ud_mumu", "eps_ud_tautau", "eps_ud_emu", "eps_ud_etau", "eps_ud_mutau"]
    parameters_18d = ["eps_e_ee", "eps_e_mumu", "eps_e_tautau", "eps_e_emu", "eps_e_etau", "eps_e_mutau",
                      "eps_u_ee", "eps_u_mumu", "eps_u_tautau", "eps_u_emu", "eps_u_etau", "eps_u_mutau",
                      "eps_d_ee", "eps_d_mumu", "eps_d_tautau", "eps_d_emu", "eps_d_etau", "eps_d_mutau"]
    n_params = len(parameters_18d)

    file_string = "dune"
    text_string = "multinest/" + file_string + "/" + file_string
    json_string = "multinest/" + file_string + "/params.json"


    # Run MultiNest.
    pymultinest.run(LogLikelihood, PriorFlow18DEmpirical, n_params,
                    outputfiles_basename=text_string,resume=True, verbose=True,
                    n_live_points=8000, evidence_tolerance=5, sampling_efficiency=0.8)
    json.dump(parameters_18d, open(json_string, 'w'))  # save parameter names
    print("Saving to: \n" + text_string)





if __name__ == "__main__":
    main()