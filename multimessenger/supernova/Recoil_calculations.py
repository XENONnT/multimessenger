#!/usr/bin/python
from scipy.special import spherical_jn
import numpy as np
from astropy import units as u
from snewpy.neutrino import Flavor
from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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

class TARGET:
    def __init__(self, atom, pure = False):
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
        self.masskeV = ((self.mass * c_speed**2).to(u.keV))
        self.massMeV = ((self.mass * c_speed**2).to(u.MeV))
        self.Qw = self.N - (1 - 4.0 * sin2theta) * self.Z
        self.rnval, self.term1 = self._get_vals()
        self.fluxes = None

    def _get_vals(self):
        # everything in energy units
        par_c = (1.23 * self.A ** (1. / 3) - 0.6) * u.fm
        rn = np.sqrt(par_c ** 2 + 7. / 3 * (np.pi * par_a) ** 2 - 5 * par_s ** 2)
        rnval = (rn / hbar / c_speed).to(u.keV ** -1)  # .value # rn value in keV
        Q_W = self.N - (1 - 4 * sin2theta) * self.Z
        term1 = corrGFmN * (GFnat ** 2) * self.masskeV * (Q_W ** 2) / (4 * np.pi)
        return rnval, term1

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

    def get_fluxes(self, model, neutrino_energies, force=False, leave=True):
        if self.fluxes is not None and not force:
            return None
        # get fluxes at each time and at each neutrino energy
        flux_unit = model.get_initial_spectra(1 * u.s, 100 * u.MeV)[Flavor.NU_E].unit
        _fluxes = np.zeros((len(model.time), len(neutrino_energies))) * flux_unit
        _fluxes = {f: _fluxes.copy() for f in Flavor}
        for f in tqdm(Flavor, total=len(Flavor), desc=self.name, leave=leave):
            for i, sec in tqdm(enumerate(model.time), total=len(model.time), desc=f.to_tex(), leave=False):
                _fluxes_dict = model.get_initial_spectra(sec, neutrino_energies)
                _fluxes[f][i, :] = _fluxes_dict[f]
        self.fluxes = _fluxes

    def dRdEr(self, model, neutrino_energies, recoil_energies):
        self.get_fluxes(model, neutrino_energies)
        # get rates per recoil energy after SN duration # integrate over time
        fluxes_per_Er = {f: np.trapz(self.fluxes[f], model.time, axis=0).to(1 / u.keV) for f in Flavor}
        # get cross-sections
        xsecs = self.nN_cross_section(neutrino_energies, recoil_energies)
        # Rough integration over neutrino energies
        integ = {f: (xsecs * fluxes_per_Er[f]) for f in Flavor}
        rates_per_Er = self._get_rates(integ, neutrino_energies)
        rates_per_Er = {k:v.to(u.m ** 2 / u.keV) for k,v in rates_per_Er.items()}
        return rates_per_Er

    def dRdt(self, model, neutrino_energies, recoil_energies):
        self.get_fluxes(model, neutrino_energies)
        xsecs = self.nN_cross_section(neutrino_energies, recoil_energies)
        # get rates per time, from all neutrino energies
        # integrate over recoil energies
        xsecs_integrated = np.trapz(xsecs, recoil_energies, axis=0)
        flux_x_xsec = {f: self.fluxes[f] * xsecs_integrated for f in Flavor}
        # integrate over nu energies
        rates_per_t = self._get_rates(flux_x_xsec, neutrino_energies)
        rates_per_t = {k:v.to(u.m ** 2 / u.s) for k,v in rates_per_t.items()}
        return rates_per_t

    def _get_rates(self, x, neutrino_energies):
        """ get rates scaled by the abundance
        """
        rates = {f: self.abund * np.trapz(x[f], neutrino_energies, axis=1) for f in Flavor}
        total_flux = np.zeros_like(rates[Flavor.NU_E])
        for f in Flavor:
            if not f.is_electron:
                _rate = rates[f] * 2  # x-neutrinos are for muon and tau
            else:
                _rate = rates[f]
            total_flux += _rate
        rates["Total"] = total_flux
        return rates

    def scale_fluxes(self,
                     distance=10*u.kpc):
        """ Scale fluxes based on abundances
            and distance, and number of atoms
        """
        N_Xe = 4.6e27 * u.count / u.tonne,
        scale = (N_Xe / (4 * np.pi * distance ** 2)).to(u.m ** 2)
        try:
            for f in self.fluxes.keys():
                self.fluxes[f] *= scale
            return self.fluxes
        except:
            raise NotImplementedError("fluxes does not exist\nCreate them by calling `get_fluxes()`")