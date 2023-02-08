#!/usr/bin/python
from scipy.special import spherical_jn
import numpy as np
from astropy import units as u


hbar = 1.0546e-27*u.cm**2 *u.g / u.s
c_speed = 2.99792458e10*u.cm/u.s # 299792458*u.m/u.s
m_e = 9.109e-31*u.kg

# these have the units of length
par_a = 0.52*u.fm
par_s = 0.9*u.fm

# everything in energy units
aval = (par_a / hbar / c_speed).to(u.keV ** -1)  # .value # a in keV
sval = (par_s / hbar / c_speed).to(u.keV ** -1)  # .value # s value in keV

sin2theta = 0.2386

# natural units
len_nat = (hbar/(m_e*c_speed)).decompose()
e_nat = (m_e * c_speed**2).to(u.keV)
t_nat = (len_nat / c_speed).to(u.s)

GF = 8.958e-44 * u.MeV * u.cm**3
GFnat = (GF/(hbar*c_speed)**3).to(u.keV**-2)
corrGFmN = len_nat**2 * e_nat**2

def _get_vals(A, Z, N, masskeV):
    # everything in energy units
    par_c = (1.23 * A ** (1. / 3) - 0.6) * u.fm
    rn = np.sqrt(par_c ** 2 + 7. / 3 * (np.pi * par_a) ** 2 - 5 * par_s ** 2)
    rnval = (rn / hbar / c_speed).to(u.keV ** -1)  # .value # rn value in keV
    Q_W = N - (1 - 4 * sin2theta) * Z
    term1 = corrGFmN * (GFnat ** 2) * masskeV * (Q_W ** 2) / (4 * np.pi)
    return rnval, term1

class Target:
    def __init__(self, atom, pure=False):
        """ Set a Target Atom
            Compute Recoil Energy Rates
            (a) counts integrated over recoil energies `dRatedErecoil`
                Here the flux passed is integrated over some time.
            (b) counts integrated over recoil energies at each time `dRatedErecoil2D`
                Thus it returns a 2D array i.e. dR/dErec at each time

        """
        self.target = atom
        self.name = atom["Type"]
        self.spin = atom["Spin"]
        self.abund = 1 if pure else atom["Fraction"]
        self.A = atom["MassNum"]
        self.Z = atom["AtomicNum"]
        self.N = self.A - self.Z
        # the following is in amu
        self.mass = (atom["Mass"] * u.u).decompose()  # in kg
        self.masskeV = ((self.mass * c_speed ** 2).to(u.keV))
        self.massMeV = ((self.mass * c_speed ** 2).to(u.MeV))
        self.Qw = self.N - (1 - 4.0 * sin2theta) * self.Z
        self.rnval, self.term1 = _get_vals(self.A, self.Z, self.N, self.masskeV)

    def __repr__(self):
        """Default representation of the model.
        """
        _repr = []
        for k,v in self.target.items():
            _repr += [f"|{k:.10s}| {v} |"]
        return "\n".join(_repr)

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        _repr = ["**The Target:**"]
        _repr += ['|Parameter|Value|',
                  '|:--------|:----:|']
        for k, v in self.target.items():
            _repr += [f"|{k:.10s}| {v} "]
        return "\n".join(_repr)


    def form_factor(self, Er):
        """
        Helms Form Factor
        Arguments
        ---------
        Er : Recoil energy (need units)
        Returns
        -------
        F(E_R) = 3*j_1(q*r_n) / (q*r_n) * exp(-(q*s)^2/2)

        """
        Er = Er.to(u.keV)  # keV
        q = np.sqrt(2 * self.masskeV * Er)  # keV
        # avoid zero division
        q = np.array([q]) if np.ndim(q) == 0 else q
        q[q == 0] = np.finfo(float).eps * q.unit
        qrn = q * self.rnval

        # spherical_jn doesn't accept units, feed values
        j1 = spherical_jn(1, qrn.value)  # unitless
        t1 = 3 * j1 * qrn.unit / qrn
        t2 = np.exp(-0.5 * (q * sval) ** 2)
        return t1 * t2

    def nN_cross_section(self, Enu, Er):
        """ neutrino nucleus Cross-Section
        Arguments
        ---------
        Enu : array, Neutrino energies, (needs unit)
        Er : array, Recoil energies, (needs unit)
        Returns
        -------
        dsigma / dE_r:
            neutrino-nucleus cross-section in units of m^2 / MeV
            shape is (len(Enu), len(Er))
        """
        # Avoid division by zero in energy PDF below.
        _Enu = [Enu.value]*Enu.unit if not np.ndim(Enu) else Enu
        _Er = [Er.value]*Er.unit if not np.ndim(Er) else Er
        _Enu[_Enu == 0] = np.finfo(float).eps * _Enu.unit
        _Er[_Er == 0] = np.finfo(float).eps * _Er.unit

        ffactor = self.form_factor(_Er) ** 2 * self.term1
        xsec = ffactor[:, np.newaxis] * (1 - 0.5 * self.masskeV * np.outer(_Er, _Enu ** -2.))
        xsec[xsec < 0] = 0
        return xsec
