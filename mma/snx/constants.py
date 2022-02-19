#!/usr/bin/python


import astropy.units as u

# # Fermi's const
# Gf = 1.1663787e-5 / u.GeV**2
# # Mixing angle
sin2the = 0.2223
hbar = 1.0546e-27*u.cm**2 *u.g / u.s
c_speed = 2.99792458e10*u.cm/u.s # 299792458*u.m/u.s
m_e = 9.109e-31*u.kg
N_Xe = 4.6e27*u.count/u.tonne

GF = 8.958e-44 * u.MeV * u.cm**3
GFnat = (GF/(hbar*c_speed)**3).to(u.keV**-2)

# natural unit conversions
len_nat = (hbar/(m_e*c_speed)).decompose()
e_nat = (m_e * c_speed**2).to(u.keV)
t_nat = (len_nat / c_speed).to(u.s)

corrGFmN = (len_nat**2 * e_nat**2).to(u.keV**2 * u.cm**2)
