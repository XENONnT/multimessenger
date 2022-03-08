#!/usr/bin/python
from scipy.special import spherical_jn
from scipy.integrate import quad, trapz
import astropy.units as u
import numpy as np

# constants
c_speed = 2.99792458e10*u.cm/u.s # 299792458*u.m/u.s
sin2the = 0.2223
hbar = 1.0546e-27*u.cm**2 *u.g / u.s
GF = 8.958e-44 * u.MeV * u.cm**3
GFnat = (GF/(hbar*c_speed)**3).to(u.keV**-2)
m_e = 9.109e-31*u.kg

# natural unit conversions
len_nat = (hbar/(m_e*c_speed)).decompose()
e_nat = (m_e * c_speed**2).to(u.keV)
t_nat = (len_nat / c_speed).to(u.s)

corrGFmN = (len_nat**2 * e_nat**2).to(u.keV**2 * u.cm**2)



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
        self.spin = atom["Spin"]
        self.abund = 1 if pure else atom["Fraction"]
        self.A = atom["MassNum"]
        self.Z = atom["AtomicNum"]
        self.N = self.A - self.Z
        # the following is in amu
        self.mass = (atom["Mass"] * u.u).decompose()  # in kg
        self.masskeV = ((self.mass * c_speed**2).to(u.keV)).value
        self.massMeV = ((self.mass * c_speed**2).to(u.MeV)).value
        self.Qw = self.N - (1 - 4.0 * sin2the) * self.Z
        self.dRatedErecoil_vect = np.vectorize(self.dRatedErecoil)
        self.dRatedErecoil2D_vect = np.vectorize(self.dRatedErecoil2D)
    
    def TransMoment(self, Er):
        """ mass in keV, Er in keV, return in keV

        """
        return np.sqrt(2 * self.masskeV * Er)
    
    def MinNeutrinoEnergy(self, Er):
        """ Er in keV, mass in keV, return in keV

        """
        return (Er + np.sqrt(Er**2 + 2 * self.masskeV * Er)) / 2.0
    
    def MaxRecoilEnergy(self, Ev):
        """ Ev in MeV, convert massMeV to MeV, return in MeV

        """
        return 2 * Ev**2 / (self.massMeV + 2 * Ev)
    
    def FormFactor(self, q):
        """ q in keV, other params converted such that
            the units of the eq is reasonable
            returns unitless form factor

        """
        if q == 0:
            return 1.0
        a = 0.52 * u.fm
        c = (1.23 * self.A**(1.0/3.0) - 0.6) * u.fm
        s = 0.9 * u.fm
        ### Natural unit conversions from lengths -> 1/energy
        aval = (a/hbar/c_speed).to(u.keV**-1).value # a in keV^-1
        cval = (c/hbar/c_speed).to(u.keV**-1).value # c in keV^-1
        sval = (s/hbar/c_speed).to(u.keV**-1).value # s in keV^-1
        
        rnval = np.sqrt(cval**2 + 7./3 * (np.pi*aval)**2 - 5*sval**2) # rn in keV^-1
        qval = q #.value # q in keV
    
        qrn = qval * rnval   # q*rn UNITLESS
        return (3 * spherical_jn(1, qrn) / qrn)**2 * np.exp(- (qval * sval)**2)

    def dXsecdErecoil(self, Er, Ev):
        """ Er in keV, Ev in MeV
            Gf^2 * mass term needed a correction
            returns Area / Energy (cm^2/keV)

        """
        # if Er.to(u.MeV).value > self.MaxRecoilEnergy(Ev).value: # MeV >< Mev
        Er_mev = Er * 1e-3
        if Er_mev > self.MaxRecoilEnergy(Ev): # MeV >< Mev
            return 0 #* u.cm**2 * u.keV**-1
        q = self.TransMoment(Er) # in keV

        GF_val = GFnat.value          # 1/keV^2
        corr_val = corrGFmN.value     # keV^2 cm^2
        first_term = GF_val**2 * self.masskeV * corr_val * (self.Qw**2) / 4. / np.pi # units = cm^2 / keV
        Ev2_kev = (Ev*1e3) **2
        sec_term = 1 - (self.masskeV * Er / 2 / Ev2_kev)   # unitless
        result = first_term * sec_term * self.FormFactor(q)**2
        return result
    
    def Integrator(self, func, xmin, xmax, **kw):
        result, foo = quad(func, xmin, xmax, **kw)
        return result
    
    def dRatedErecoil(self, Er, Flux):
        """ Flux: function to return flux in units of counts/(MeV*cm^2) - integrated over 10sec
            dXsecdErecoil returns cm^2 / keV -> converted into cm^2/MeV
            so that `dRatedErEv` returns counts (unitless)
            returns counts integrated over recoil energies

        """
        def dRatedErdEv(_ev):
            Flux_at_Ev = Flux(_ev)
            dXsec_dEr = self.dXsecdErecoil(Er, _ev) * 1e3 # from cm^2/keV => cm^2/MeV
            # return (dXsec_dEr.to(u.cm**2/u.MeV)).value * Flux_at_Ev
            return dXsec_dEr * Flux_at_Ev                 # cm^2/MeV  *  counts/(MeV*cm^2) = cts/MeV^2

        evmin = self.MinNeutrinoEnergy(Er)*1e-3     # from keV -> MeV
        return self.Integrator(dRatedErdEv, evmin, 30) * self.abund / (self.target["Mass"])

    def dRatedErecoil2D(self, Er, Flux, t):
        """ This time the flux should be sampled from a 2D surface

        """
        def dRatedErdEv(_ev, t):
            """
            Flux[t] : interpolator at t
            Flux[t](_ev) : interpolated count at t time, and _ev neutrino energy
            
            """
            Flux_at_Ev = Flux[t](_ev)
            if Flux_at_Ev < 0:
                Flux_at_Ev = 0
            dXsec_dEr = self.dXsecdErecoil(Er, _ev) * 1e3
            return dXsec_dEr * Flux_at_Ev

        evmin = self.MinNeutrinoEnergy(Er)*1e-3
        return self.Integrator(dRatedErdEv, evmin, 30, args=t) * self.abund / (self.target["Mass"])
    