# -*- coding: utf-8 -*-

"""
Helper functions to define the observables. e.g.:

 - Includes the calculation of the neutrino pZ in the same way as done in the ATLAS VH(bb) analyses

 - Includes the calculation of a set of angular observables.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import numpy as np
from math import sqrt, fabs

import os

import argparse as ap

from madminer.lhe import LHEReader
from madminer.utils.particle import MadMinerParticle
import vector

# "MET" set of observables
# Elements in the lists will be given as input to the add_observable function
observable_names = [
    'b1_px', 'b1_py', 'b1_pz', 'b1_e',
    'b2_px', 'b2_py', 'b2_pz', 'b2_e',
    'l_px', 'l_py', 'l_pz', 'l_e',
    'v_px', 'v_py',
    'pt_b1', 'pt_b2', 'pt_l', 'met', 'pt_w', 'pt_h',
    'eta_b1', 'eta_b2', 'eta_l', 'eta_h',
    'phi_b1', 'phi_b2', 'phi_l', 'phi_v', 'phi_w', 'phi_h',
    'theta_b1', 'theta_b2', 'theta_l', 'theta_h',
    'dphi_bb', 'dphi_lv', 'dphi_wh',
    'm_bb', 'mt_lv', 'mt_tot',
    'q_l',
    'dphi_lb1', 'dphi_lb2', 'dphi_vb1', 'dphi_vb2',
    'dR_bb', 'dR_lb1', 'dR_lb2'
]

list_of_observables = [
    'j[0].px', 'j[0].py', 'j[0].pz', 'j[0].e',
    'j[1].px', 'j[1].py', 'j[1].pz', 'j[1].e',
    'l[0].px', 'l[0].py', 'l[0].pz', 'l[0].e',
    'met.px', 'met.py',   
    'j[0].pt', 'j[1].pt', 'l[0].pt', 'met.pt', '(l[0] + met).pt', '(j[0] + j[1]).pt',
    'j[0].eta', 'j[1].eta', 'l[0].eta', '(j[0] + j[1]).eta',
    'j[0].phi', 'j[1].phi', 'l[0].phi', 'met.phi', '(l[0] + met).phi', '(j[0] + j[1]).phi',
    'j[0].theta', 'j[1].theta', 'l[0].theta', '(j[0] + j[1]).theta',
    'j[0].deltaphi(j[1])', 'l[0].deltaphi(met)', '(l[0] + met).deltaphi(j[0] + j[1])',
    '(j[0] + j[1]).m', '(l[0] + met).mt', '(j[0] + j[1] + l[0] + met).mt',
    'l[0].charge',
    'l[0].deltaphi(j[0])', 'l[0].deltaphi(j[1])', 'met.deltaphi(j[0])', 'met.deltaphi(j[1])',
    'j[0].deltaR(j[1])', 'l[0].deltaR(j[0])', 'l[0].deltaR(j[1])'
]

# solves quadratic equation to get two possible solutions for pz_nu
# calculation similar to what is done in the ATLAS VHbb analyses (getNeutrinoPz function in AnalysisReader.cxx)
def get_neutrino_pz(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  if (met is None) or leptons==[] or jets==[]:
      raise TypeError("one of the required inputs of wrong type (not found)")
  
  # W boson mass (GeV)
  m_w=80.379
  
  pz_nu_1=0.0
  pz_nu_2=0.0

  if debug:
    logging.debug(f'leading lepton 4-vector: {leptons[0]}')
    logging.debug(f'leading jet 4-vector: {jets[0]}')
    logging.debug(f'subleading jet 4-vector: {jets[1]}')
    logging.debug(f'met 4-vector: {met}')

  # auxiliary quantities
  mu = 0.5 * pow(m_w,2) + (leptons[0].px*met.px + leptons[0].py*met.py) # = tmp/2
  delta = pow(mu*leptons[0].pz / pow(leptons[0].pt,2) ,2) - (pow(leptons[0].e,2)*pow(met.pt,2) - pow(mu,2))/pow(leptons[0].pt,2) 

  # neglecting any imaginary parts (equivalent to setting mtw = mw)
  if delta < 0.0:
    delta=0.0
  
  pz_nu_1=mu*leptons[0].pz/pow(leptons[0].pt,2) + sqrt(delta)
  pz_nu_2=mu*leptons[0].pz/pow(leptons[0].pt,2) - sqrt(delta)

  # NOTE: MadMinerParticle inherits from Vector class in scikit-hep)
  nu_1_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu_1,m=0.0)
  nu_2_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu_2,m=0.0)

  nu_1=MadMinerParticle(nu_1_vec.azimuthal,nu_1_vec.longitudinal,nu_1_vec.temporal)
  nu_2=MadMinerParticle(nu_2_vec.azimuthal,nu_2_vec.longitudinal,nu_2_vec.temporal)

  if debug:
    logging.debug(f'nu_1 4-vector: {nu_1}; nu_2 4-vector: {nu_2}')

  h_candidate=jets[0]+jets[1]

  w_candidate_1=leptons[0]+nu_1
  w_candidate_2=leptons[0]+nu_2

  # correct definition of betaZ, but neutrino pZ reconstruction studies in NLO SM sample showed that this is not optimal
  """
  h_candidate_betaZ=h_candidate.to_beta3().z

  w_candidate_1_betaZ=w_candidate_1.to_beta3().z
  w_candidate_2_betaZ=w_candidate_2.to_beta3().z
  """
  h_candidate_betaZ=h_candidate.pz/np.sqrt(pow(h_candidate.pz,2)+h_candidate.M2) 
  w_candidate_1_betaZ=w_candidate_1.pz/np.sqrt(pow(w_candidate_1.pz,2)+w_candidate_1.M2) 
  w_candidate_2_betaZ=w_candidate_2.pz/np.sqrt(pow(w_candidate_2.pz,2)+w_candidate_2.M2) 

  
  if debug:
    logging.debug(f'h_candidate 4-vector: {h_candidate}')
    logging.debug(f'w_candidate_1 4-vector: {w_candidate_1}; w_candidate_2 4-vector: {w_candidate_2}')

    logging.debug(f'h_candidate_BetaZ: {h_candidate_betaZ}')

    logging.debug(f'w_candidate_1_BetaZ: {w_candidate_1_betaZ}')
    logging.debug(f'w_candidate_2_BetaZ: {w_candidate_2_betaZ}')
  

  dBetaZ_1=fabs(w_candidate_1_betaZ - h_candidate_betaZ)
  dBetaZ_2=fabs(w_candidate_2_betaZ - h_candidate_betaZ)

  if debug:
    logging.debug(f'dBetaZ_1: {dBetaZ_1}; dBetaZ_2: {dBetaZ_2}')

  if dBetaZ_1 <= dBetaZ_2:
    if debug:
      logging.debug(f'pz_nu: {pz_nu_1}')
    return pz_nu_1
  else:
    if debug:
      print(f'pz_nu: {pz_nu_2}')
    return pz_nu_2

def get_cos_thetaStar(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu
  # equivalent to boostCM_of_p4(w_candidate), which hasn't been implemented yet for MomentumObject4D
  # negating spatial part only to boost *into* the CM, default would be to boost *away* from the CM
  lead_lepton_w_centerOfMass=leptons[0].boost_beta3(-w_candidate.to_beta3())

  if debug:

    logging.debug(f'W candidate 4-vector: {w_candidate}')
    w_candidate_w_centerOfMass=w_candidate.boost_beta3(-w_candidate.to_beta3())
    print(f'W candidate 4-vector boosted to the CM of the W candidate (expect[0.0,0.0,0.0,m_w]): {w_candidate_w_centerOfMass}')
    print(f'leading lepton 4-vector boosted to the CM of the W candidate: {lead_lepton_w_centerOfMass}')  
  
  cos_thetaStar=lead_lepton_w_centerOfMass.to_Vector3D().dot(w_candidate.to_Vector3D())/(fabs(lead_lepton_w_centerOfMass.p) * fabs(w_candidate.p))

  if debug:
    print(f'cos_thetaStar= {cos_thetaStar}')

  return cos_thetaStar

def get_ql_cos_thetaStar(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  cos_thetaStar=get_cos_thetaStar(leptons=leptons,jets=jets,met=met)

  if debug:
    logging.debug(f'ql_cos_thetaStar = {leptons[0].charge * cos_thetaStar}')
  return leptons[0].charge * cos_thetaStar

def get_cos_deltaPlus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):
  
  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu
  # equivalent to boostCM_of_p4(w_candidate), which hasn't been implemented yet for MomentumObject4D
  # negating spatial part only to boost *into* the CM, default would be to boost *away* from the CM
  lead_lepton_w_centerOfMass=leptons[0].boost_beta3(-w_candidate.to_beta3())

  if debug:
    logging.debug(f'W candidate 4-vector: {w_candidate}')
    w_candidate_w_centerOfMass=w_candidate.boost_beta3(-w_candidate.to_beta3())
    logging.debug(f'W candidate 4-vector boosted to the CM of the W candidate (expect[0.0,0.0,0.0,m_w]): {w_candidate_w_centerOfMass}')
    logging.debug(f'leading lepton 4-vector boosted to the CM of the W candidate: {lead_lepton_w_centerOfMass}')
  
  h_candidate=jets[0]+jets[1]

  h_cross_w=h_candidate.to_Vector3D().cross(w_candidate.to_Vector3D())

  cos_deltaPlus=lead_lepton_w_centerOfMass.to_Vector3D().dot(h_cross_w)/(fabs(lead_lepton_w_centerOfMass.p) * fabs(h_cross_w.mag))

  if debug:
    logging.debug(f'cos_deltaPlus = {cos_deltaPlus}')

  return cos_deltaPlus

def get_ql_cos_deltaPlus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  cos_deltaPlus=get_cos_deltaPlus(leptons=leptons,jets=jets,met=met)
  if debug:
    logging.debug(f'ql_cos_deltaPlus = {leptons[0].charge * cos_deltaPlus}')
  return leptons[0].charge * cos_deltaPlus

def get_cos_deltaMinus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):

  pz_nu=get_neutrino_pz(leptons=leptons,jets=jets,met=met,debug=debug)

  nu_vec = vector.obj(px=met.px,py=met.py,pz=pz_nu,m=0.0)
  nu=MadMinerParticle(nu_vec.azimuthal,nu_vec.longitudinal,nu_vec.temporal)

  w_candidate=leptons[0]+nu

  h_candidate=jets[0]+jets[1]

  if debug:

    logging.debug(f'H candidate 4-vector: {h_candidate}')
    h_candidate_h_centerOfMass=h_candidate.boost_beta3(-h_candidate.to_beta3())
    logging.debug(f'H candidate 4-vector boosted to the CM of the H candidate (expect[0.0,0.0,0.0,m_H]): {h_candidate_h_centerOfMass}')
    
  lead_lepton_inv_h_centerOfMass=leptons[0].boost_beta3(h_candidate.to_beta3())
  nu_inv_h_centerOfMass=nu.boost_beta3(h_candidate.to_beta3())

  if debug:  
    logging.debug(f'leading lepton 4-vector boosted to the CM of the H candidate with an inverted momentum: {lead_lepton_inv_h_centerOfMass}')
    logging.debug(f'neutrino 4-vector boosted to the CM of the H candidate with an inverted momentum: {nu_inv_h_centerOfMass}')

  lep_cross_nu_inv_h_centerOfMass=lead_lepton_inv_h_centerOfMass.to_Vector3D().cross(nu_inv_h_centerOfMass.to_Vector3D())

  cos_deltaMinus=lep_cross_nu_inv_h_centerOfMass.dot(w_candidate.to_Vector3D())/(fabs(lep_cross_nu_inv_h_centerOfMass.mag)*fabs(w_candidate.p))

  if debug:
    logging.debug(f'cos_deltaMinus = {cos_deltaMinus}')

  return cos_deltaMinus

def get_ql_cos_deltaMinus(particles=[],leptons=[],photons=[],jets=[],met=None,debug=False):
  cos_deltaMinus=get_cos_deltaMinus(leptons=leptons,jets=jets,met=met)
  if debug:
    logging.debug(f'ql_cos_deltaMinus = {leptons[0].charge * cos_deltaMinus}')
  return leptons[0].charge * cos_deltaMinus

