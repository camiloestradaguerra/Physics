__name__ = "Utilities for Adiabatic entanglement in two-atom cavity QED code"

import math
import numpy as np
import matplotlib.pyplot as plt


def gauss_prof(x, x0, E0):
  """
  Gaussian profile of the electromagnetic field inside of the QED cavity

  :param x: x position in the cavity (array)
  :param x0: value of the variance of the gaussian profile (scalar)
  """
  EM_f = E0*np.exp(-x**2/(4*x0**2))
  return EM_f

def coupl(par, g, tau, delta):
  """
  Coupling constant betwenn atoms and EM field
  
  :param par: 1 if n1 or 2 if n2
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return n1 or n2: coupling constants
  """
  if par == 1:
    n1 = g*np.exp(-(tau+delta)**2)
    return n1
  if par == 2:
    n2 = g*np.exp(-(tau-delta)**2)
    return n2

def f_func(n, g, tau, delta):
  """
  Coupling constant betwenn atoms and EM field
  
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return f: complementary function
  """
  f = np.sqrt((coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)**2+16*(n+1)*(n+2)*coupl(1, g, tau, delta)**2*coupl(2, g, tau, delta)**2)
  return f

def energy_m(n, g, tau, delta):
  """
  E_m

  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return e_m: E_minus
  """
  e_m = np.sqrt(0.5)*np.sqrt((3+2*n)*(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)-f_func(n, g, tau, delta))
  return e_m

def energy_p(n, g, tau, delta):
  """
  E_p

  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return e_p: E_plus
  """
  e_p = np.sqrt(0.5)*np.sqrt((3+2*n)*(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)+f_func(n, g, tau, delta))
  return e_p

def a_coef(par, n, g, tau, delta):
  """
  Coefficient A_minus and A_plus for the eigenstates

  :param par: 'm' for A_minus or 'p' for A_plus
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return am or ap: A_plus or A_minus coeff.
  """
  a_num = np.sqrt(4*(n+1)*(n+2)*coupl(1, g, tau, delta)**2*coupl(2, g, tau, delta)**2)
  if par == 'p':
    ap_den = np.sqrt(f_func(n, g, tau, delta)**2+(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)*f_func(n, g, tau, delta))
    ap = a_num/ap_den
    return ap
  if par == 'm':
    am_den = np.sqrt(f_func(n, g, tau, delta)**2-(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)*f_func(n, g, tau, delta))
    am = -a_num/am_den
    return am
  
def b_coef(par, n, g, tau, delta):
  """
  Coefficient B_minus and B_plus for the eigenstates

  :param par: 'm' for B_minus or 'p' for B_plus
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return bp or bm: B_plus or B_minus coeff.
  """
  if par == 'p':
    bp_num = d_coef('p', n, g, tau, delta)*energy_p(n, g, tau, delta)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2-f_func(n, g, tau, delta))
    bp_den = 2*coupl(2, g, tau, delta)*np.sqrt(n+2)*(energy_p(n, g, tau, delta)**2+(n+1)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2))
    bp = bp_num/bp_den
    return bp
  if par == 'm':
    bm_num = d_coef('m', n, g, tau, delta)*energy_m(n, g, tau, delta)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2+f_func(n, g, tau, delta))
    bm_den = 2*coupl(2, g, tau, delta)*np.sqrt(n+2)*(energy_m(n, g, tau, delta)**2+(n+1)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2))
    bm = bm_num/bm_den
    return bm

def c_coef(par, n, g, tau, delta):
  """
  Coefficient C_minus and C_plus for the eigenstates

  :param par: 'm' for C_minus or 'p' for C_plus
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return cp or cm: C_plus or C_minus coeff.
  """
  if par == 'p':
    cp_num = d_coef('p', n, g, tau, delta)*energy_p(n, g, tau, delta)*coupl(1, g, tau, delta)*(3+2*n)
    cp_den = np.sqrt(n+2)*(energy_p(n, g, tau, delta)**2+(n+1)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2))
    cp = cp_num/cp_den
    return cp
  if par == 'm':
    cm_num = d_coef('m', n, g, tau, delta)*energy_m(n, g, tau, delta)*coupl(1, g, tau, delta)*(3+2*n)
    cm_den = np.sqrt(n+2)*(energy_m(n, g, tau, delta)**2+(n+1)*(coupl(1, g, tau, delta)**2-coupl(2, g, tau, delta)**2))
    cm = cm_num/cm_den
    return cm

def d_coef(par, n, g, tau, delta):
  """
  Coefficient B_minus and B_plus for the eigenstates

  :param par: 'm' for B_minus or 'p' for B_plus
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return cp or cm: C_plus or C_minus coeff
  """
  if par == 'p':
    dp = 0.5*np.sqrt(1+(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)/f_func(n, g, tau, delta))
    return dp
  if par == 'm':
    dm = 0.5*np.sqrt(1-(coupl(1, g, tau, delta)**2+coupl(2, g, tau, delta)**2)/f_func(n, g, tau, delta))
    return dm 

def psi(num, n, g, tau, delta):
  """
  Calculates the eigenstates of the Hamiltonian 
  
  :param num: number of the eigenstate
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return psi: eigenstate
  """
  if num == 1:
    psi = np.array([a_coef('m', n, g, tau, delta), b_coef('m', n, g, tau, delta), -c_coef('m', n, g, tau, delta), d_coef('m', n, g, tau, delta)])
    return psi
  if num == 2:
    psi = np.array([a_coef('m', n, g, tau, delta), -b_coef('m', n, g, tau, delta), +c_coef('m', n, g, tau, delta), d_coef('m', n, g, tau, delta)])
    return psi
  if num == 3:
    psi = np.array([a_coef('p', n, g, tau, delta), b_coef('p', n, g, tau, delta), -c_coef('p', n, g, tau, delta), d_coef('p', n, g, tau, delta)])
    return psi
  if num == 4:
    psi = np.array([a_coef('p', n, g, tau, delta), -b_coef('p', n, g, tau, delta), +c_coef('p', n, g, tau, delta), d_coef('p', n, g, tau, delta)])
    return psi

def ad_psi(num, par, n, g, tau, delta):
  """
  Calculates the adiabatic eigenstates of the Hamiltonian 
  
  :param num: number of the eigenstate
  :param par: if par '-' for  tau -> -inf and '+' for tau -> +inf
  :param n: number of photons
  :param g: g factor
  :param tau: dimensionless time
  :param delta: delay between atoms
  :return psi: eigenstate
  """
  if num == 1 and par == '-':
    ad_psi =  -np.array([1, -1, 0, 0])/np.sqrt(2)
    return ad_psi
  if num == 1 and par == '+':
    ad_psi =  -np.array([1, 0, -1, 0])/np.sqrt(2)
    return ad_psi
  if num == 2 and par == '-':
    ad_psi =  -np.array([1, 1, 0, 0])/np.sqrt(2)
    return ad_psi
  if num == 2 and par == '+':
    ad_psi =  -np.array([1, 0, 1, 0])/np.sqrt(2)
    return ad_psi
  if num == 3 and par == '-':
    ad_psi =  -np.array([0, 0, -1, 1])/np.sqrt(2)
    return ad_psi
  if num == 3 and par == '+':
    ad_psi =  -np.array([0, -1, 0, 1])/np.sqrt(2)
    return ad_psi
  if num == 4 and par == '-':
    ad_psi =  -np.array([0, 0, 1, 1])/np.sqrt(2)
    return ad_psi
  if num == 4 and par == '+':
    ad_psi =  -np.array([0, 1, 0, 1])/np.sqrt(2)
    return ad_psi