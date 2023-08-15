#!/usr/bin/python

# Reference: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=Ar
# J. S. Coursey, D. J. Schwab, J. J. Tsai, and R. A. Dragoset, 
# Atomic Weights and Isotopic Compositions with Relative Atomic Masses, 
# NIST Physical Measurement Laboratory

Ar36 = {
    'Type'		: 'Ar36',
    'MassNum'	: 36,
    'AtomicNum'	: 18,
    'Mass'		: 35.967545105,
    'Spin'		: 0,
    'Fraction'	: 3.336e-3
}

Ar38 = {
    'Type'		: 'Ar38',
    'MassNum'	: 38,
    'AtomicNum'	: 18,
    'Mass'		: 37.962732,
    'Spin'		: 0,
    'Fraction'	: 6.29E-4
}

Ar40 = {
    'Type'		: 'Ar40',
    'MassNum'	: 40,
    'AtomicNum'	: 18,
    'Mass'		: 39.962383,
    'Spin'		: 0,
    'Fraction'	: 0.996035
}

ATOM_TABLE = {
    'Ar36'	: Ar36,
    'Ar38'	: Ar38,
    'Ar40'	: Ar40,
}
