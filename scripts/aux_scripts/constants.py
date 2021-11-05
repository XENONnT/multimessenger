#!/usr/bin/python

Xe124 = {
    'Type'		: 'Xe124',
    'MassNum'	: 124,
    'AtomicNum'	: 54,
    'Mass'		: 123.905893,
    'Spin'		: 0,
    'Fraction'	: 9.52E-4
}

Xe126 = {
    'Type'		: 'Xe126',
    'MassNum'	: 126,
    'AtomicNum'	: 54,
    'Mass'		: 125.904274,
    'Spin'		: 0,
    'Fraction'	: 8.90E-4
}

Xe128 = {
    'Type'		: 'Xe128',
    'MassNum'	: 128,
    'AtomicNum'	: 54,
    'Mass'		: 127.9035313,
    'Spin'		: 0,
    'Fraction'	: 0.019102
}

Xe129 = {
    'Type'		: 'Xe129',
    'MassNum'	: 129,
    'AtomicNum'	: 54,
    'Mass'		: 128.9047794,
    'Spin'		: 0.5,
    'Fraction'	: 0.264006
}

Xe130 = {
    'Type'		: 'Xe130',
    'MassNum'	: 130,
    'AtomicNum'	: 54,
    'Mass'		: 129.9035080,
    'Spin'		: 0,
    'Fraction'	: 0.040710
}

Xe131 = {
    'Type'		: 'Xe131',
    'MassNum'	: 131,
    'AtomicNum'	: 54,
    'Mass'		: 130.9050824,
    'Spin'		: 1.5,
    'Fraction'	: 0.212324
}

Xe132 = {
    'Type'		: 'Xe132',
    'MassNum'	: 132,
    'AtomicNum'	: 54,
    'Mass'		: 131.9041535,
    'Spin'		: 0,
    'Fraction'	: 0.269086
}

Xe134 = {
    'Type'		: 'Xe134',
    'MassNum'	: 134,
    'AtomicNum'	: 54,
    'Mass'		: 133.9053945,
    'Spin'		: 0,
    'Fraction'	: 0.104357
}

Xe136 = {
    'Type'		: 'Xe136',
    'MassNum'	: 136,
    'AtomicNum'	: 54,
    'Mass'		: 135.907219,
    'Spin'		: 0,
    'Fraction'	: 0.088573
}

ATOM_TABLE = {
    'Xe124'	: Xe124,
    'Xe126'	: Xe126,
    'Xe128'	: Xe128,
    'Xe129'	: Xe129,
    'Xe130'	: Xe130,
    'Xe131'	: Xe131,
    'Xe132'	: Xe132,
    'Xe134'	: Xe134,
    'Xe136'	: Xe136
}

import astropy.units as u

# # Fermi's const
# Gf = 1.1663787e-5 / u.GeV**2
# # Mixing angle
sin2the = 0.2223

# c_speed = 299792458*u.m/u.s
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