import sys
import json
import numpy as np
from numpy import log, exp, pi, zeros, genfromtxt
import matplotlib.pyplot as plt
import pandas as pd

import pymultinest
import mpi4py

from rebin_flux import RebinAtmosphericFlux

from pyCEvNS.events import*
from pyCEvNS.oscillation import*
from pyCEvNS.plot import CrediblePlot


# Modulate the atmospheric neutrino flux by the survival and transition probabilities.
def ModulateFlux(params, flux_array, appearance_flavor='tau'):
  nsi = NSIparameters(0)
  #nsi.epe = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
   #          'mt': params[5]}
  nsi.epu = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
             'mt': params[5]}
  nsi.epd = {'ee': params[0], 'mm': params[1], 'tt': params[2], 'em': params[3], 'et': params[4],
             'mt': params[5]}

  entries = flux_array.shape[0]
  flux_mod = np.zeros([flux_array.shape[0], flux_array.shape[1]])

  for n in range(0, entries):
    this_e = flux_array[n, 0]
    this_z = flux_array[n, 1]

    # Calculate tau survival and transition probability densities.
    if appearance_flavor == 'tau':
        tau_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='tau', nuf='tau')
        e_tau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                    op=oscillation_parameters(delta=params[6]), nui='e', nuf='tau')
        mu_tau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                     op=oscillation_parameters(delta=params[6]), nui='mu',nuf='tau')
        flux_mod[n, 4] = (tau_surv * flux_array[n, 4]) \
                           + (e_tau_trans * flux_array[n, 2]) \
                           + (mu_tau_trans * flux_array[n, 3])

    if appearance_flavor == 'mu':
        mu_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                op=oscillation_parameters(delta=params[6]), nui='mu', nuf='mu')
        e_mu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='e', nuf='mu')
        tau_mu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                     op=oscillation_parameters(delta=params[6]), nui='tau', nuf='mu')
        flux_mod[n, 3] = (mu_surv * flux_array[n, 3]) \
                           + (e_mu_trans * flux_array[n, 2]) \
                           + (tau_mu_trans * flux_array[n, 4])

    if appearance_flavor == 'e':
        e_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                               op=oscillation_parameters(delta=params[6]), nui='e', nuf='e')
        mu_e_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='mu', nuf='e')
        tau_e_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                    op=oscillation_parameters(delta=params[6]), nui='tau', nuf='e')
        flux_mod[n, 2] = (e_surv * flux_array[n, 2]) \
                           + (mu_e_trans * flux_array[n, 3]) \
                           + (tau_e_trans * flux_array[n, 4])

    if appearance_flavor == 'taubar':  # Integrate all flavors for DD experiments
        atau_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                 op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='taubar')
        ae_atau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                    op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='taubar')
        amu_atau_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                     op=oscillation_parameters(delta=params[6]), nui='mubar',nuf='taubar')
        flux_mod[n, 7] = (atau_surv * flux_array[n, 7]) \
                           + (ae_atau_trans * flux_array[n, 5]) \
                           + (amu_atau_trans * flux_array[n, 6])

    if appearance_flavor == 'mubar':
        amu_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                op=oscillation_parameters(delta=params[6]), nui='mubar', nuf='mubar')
        ae_amu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='mubar')
        atau_amu_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                     op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='mubar')
        flux_mod[n, 6] = (amu_surv * flux_array[n, 6]) \
                           + (ae_amu_trans * flux_array[n, 5]) \
                           + (atau_amu_trans * flux_array[n, 7])

    if appearance_flavor == 'ebar':
        ae_surv = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                               op=oscillation_parameters(delta=params[6]), nui='ebar', nuf='ebar')
        amu_ae_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                   op=oscillation_parameters(delta=params[6]), nui='mubar', nuf='ebar')
        atau_ae_trans = survial_atmos(this_e, zenith=this_z, epsi=nsi,
                                    op=oscillation_parameters(delta=params[6]), nui='taubar', nuf='ebar')
        flux_mod[n, 5] = (ae_surv * flux_array[n, 5]) \
                           + (amu_ae_trans * flux_array[n, 6]) \
                           + (atau_ae_trans * flux_array[n, 7])


  flux_mod[:,0] = flux_array[:,0]
  flux_mod[:,1] = flux_array[:,1]
  return flux_mod



# Take in an nsi and return # of events integrated over energy and zenith
def EventsGenerator(nsi_array, df, expo, appear_flavor):
  det = Detector("dune")
  nsteps_e = 19
  nsteps_z = 20
  dcos_th = 0.05
  e_high = 1000
  e_low = 100

  flux_mod = ModulateFlux(nsi_array, df, appear_flavor)

  nsi = NSIparameters(0)
  nsi.epu = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2], 'em': nsi_array[3], 'et': nsi_array[4],
             'mt': nsi_array[5]}
  nsi.epd = {'ee': nsi_array[0], 'mm': nsi_array[1], 'tt': nsi_array[2], 'em': nsi_array[3], 'et': nsi_array[4],
             'mt': nsi_array[5]}

  energy_arr = np.linspace(e_low, e_high, nsteps_e)
  zenith_arr = np.unique(flux_mod[:, 1])

  observed_events = np.zeros((nsteps_e-1)*nsteps_z)
  this_obs = 0

  for i in range(0, nsteps_z):
    zenith_bin = np.delete(flux_mod[flux_mod[:,1] == zenith_arr[i]], [1], axis=1)
    e_a = energy_arr[0]
    def convolveFlux(ev,nui,nuf):
      return survial_atmos(ev, zenith=zenith_arr[i], epsi=nsi,
                           op=oscillation_parameters(delta=nsi_array[6]),
                           nui=nui, nuf=nuf)

    for j in range (1, nsteps_e):
      e_b = energy_arr[j]
      observed_events[this_obs] = 2*pi*dcos_th*binned_events_ccqe(e_a, e_b, expo, det, Flux(zenith_bin),
                                                                  flavor=appear_flavor)
      #pois_error = np.sqrt(observed_events[this_obs])
      #smear = np.random.random_integers(-0.5 * pois_error, 0.5 * pois_error)
      observed_events[this_obs] = np.round(observed_events[this_obs]) #+ smear
      #if observed_events[this_obs] < 0:
      #  observed_events[this_obs] = 0
      this_obs += 1

      # Iterate left edge
      e_a = e_b

  return observed_events




# Map the interval [0,1] to the interval [eps_min, eps_max] as a flat prior for each NSI parameter.
# Constraints from 1905.05203
def FlatPrior(cube, ndim, nparams):
  #cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee - eps_mumu
  #cube[1] = 0  # we subtract eps_mumu from the NSI matrix, eps_ee' = eps_ee - eps_mumu, same for tau.
  #cube[2] = 0.25 * (cube[2] - 0.2)  # eps_tautau - eps_mumu
  cube[0] = 0.5 * (2 * cube[0] - 1)  # eps_ee
  cube[1] = 0.5 * (2 * cube[1] - 1)  # eps_mumu
  cube[2] = 0.5 * (2 * cube[2] - 1)  # eps_tautau
  cube[3] = 0.12 * (2 * cube[3] - 1)  # eps_emu
  cube[4] = 0.3 * (2 * cube[4] - 1)  # eps_etau
  cube[5] = 0.028 * (2 * cube[5] - 1)  # eps_mutau
  cube[6] = 2 * pi * cube[6]  # delta_CP




def main(appear_flavor='tau'):
  # Set the exposure (DUNE defaults):
  kTon = 40
  years = 10
  days_per_year = 365
  kg_per_kton = 1000000

  exposure = years * days_per_year * kTon * kg_per_kton

  # Load the flux file.
  atmos = RebinAtmosphericFlux('pyCEvNS/data/atmos.txt', 20, True)

  # 2D chisquare function
  n_sm = EventsGenerator([0, 0, 0, 0, 0, 0, 1.5*pi], atmos, exposure, appear_flavor)
  width = np.sqrt(n_sm)
  def LogLikelihood(cube, D, N, df=atmos, expo=exposure):
    n_signal = EventsGenerator(cube, df, expo, appear_flavor)
    likelihood = np.ones(n_signal.shape[0])

    for i in range(n_signal.shape[0]):
      if width[i]==0:
        continue
      likelihood[i] = exp(-0.5 * ((n_signal[i] - n_sm[i]) / width[i]) ** 2) / (2 * pi * width[i] ** 2) ** 0.5
    return np.sum(np.log(likelihood))

  plot = False
  if plot == True:
      data = np.reshape(n_sm, (20,18))
      e_bins = np.linspace(100, 1000, 19)
      z_bins = np.linspace(-1.0, 0.0, 21)
      fig1, ax1 = plt.subplots()
      cmap1 = ax1.pcolormesh(e_bins, z_bins, data)
      fig1.colorbar(cmap1)
      fig1.savefig("plots/rates/dune/png/dune_atmos_event_rate_nsi.png")






  # Define model parameters
  parameters = ["eps_ee", "eps_mumu", "eps_tautau", "eps_emu", "eps_etau", "eps_mutau", "delta_cp"]
  n_params = len(parameters)

  file_string = "all_nsi_dune_realistic_" + appear_flavor
  text_string = "nsi_multinest/" + file_string + "/" + file_string
  json_string = "nsi_multinest/" + file_string + "/params.json"

  print("Saving to: \n" + text_string)

  # run Multinest
  pymultinest.run(LogLikelihood, FlatPrior, n_params, outputfiles_basename=text_string,resume=False, verbose=True,
                  n_live_points=4000, evidence_tolerance=0.5, sampling_efficiency=0.3)
  json.dump(parameters, open(json_string, 'w'))  # save parameter names



if __name__ == "__main__":
  main(str(sys.argv[1]))
