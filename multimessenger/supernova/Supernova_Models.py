#!/usr/bin/python
"""
Last Update: 31-08-2021
-----------------------
Supernova Models module.
Methods to deal with supernova lightcurve and derived properties

Author: Melih Kara kara@kit.edu

Notes
-----
uses _pickle module, check here https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
How to pickle yourself https://stackoverflow.com/questions/2709800/how-to-pickle-yourself

_get_t_int_flux -> sum or integration? (sum)

###
The recoil energies changes after 1D truncation. Thus, the next time it looks for the data, it
finds the wrong emin, emax values and can confuse.

Todo: The total_rates1D and the total rates from 2D does NOT give the same
 Need to investigate this! # might be fixed/understood

"""
import os, click
import numpy as np
import _pickle as pickle
import scipy.interpolate as itp
from .sn_utils import _inverse_transform_sampling
from .Xenon_Atom import ATOM_TABLE
import configparser
import astropy.units as u
N_Xe = 4.6e27*u.count/u.tonne

from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def get_composite():
    """ Get a Xenon nucleus composite
    """
    from .Recoil_calculations import TARGET
    XeNuc = [TARGET(ATOM_TABLE["Xe124"], pure=False),
             TARGET(ATOM_TABLE["Xe126"], pure=False),
             TARGET(ATOM_TABLE["Xe128"], pure=False),
             TARGET(ATOM_TABLE["Xe129"], pure=False),
             TARGET(ATOM_TABLE["Xe130"], pure=False),
             TARGET(ATOM_TABLE["Xe131"], pure=False),
             TARGET(ATOM_TABLE["Xe132"], pure=False),
             TARGET(ATOM_TABLE["Xe134"], pure=False),
             TARGET(ATOM_TABLE["Xe136"], pure=False)]
    return XeNuc

class SN_LightCurve:
    """ Deal with a given SN lightcurve
    """

    def __init__(self,
                 progenitor_mass,
                 metallicity,
                 time_of_revival,
                 distance=10,
                 recoil_energies=(0, 20, 50),
                 filename_=None,
                 storage=None,
                 force=False):
        """
        Parameters
        ----------
        progenitor_mass : float
            mass of the progenitor star. Depends on the availability of models.
        metallicity : float
            metallicity of the progenitor star.
        time_of_revival : flat
            time of the core collapse
        distance : float, optional
            Supernova distance in kpc. Deafult is 10 kpc
        recoil_energies : tuple
            recoil energies to search for interactions
            tuple with (start, stop, step)
        filename : str, optional
            Filename to save. If None, constructs a name based on properties.

        """
        self.XeNuc = get_composite()
        self.M = progenitor_mass
        self.t_revival = time_of_revival
        self.Z = metallicity
        self.dist = distance
        self.t0 = 0
        self.tf = 10
        self.name = filename_ or f'Object_M{self.M}-Z{self.Z}_dist{self.dist}.p'
        self.recoil_en = np.linspace(recoil_energies[0], recoil_energies[1], recoil_energies[2])
        if storage is None:
            try:
                # try to find from the default config
                config = configparser.ConfigParser()
                config.read('/dali/lgrandi/melih/mma/data/basic_conf.conf')
                self.storage = config['paths']['data']
            except:
                self.storage = os.getcwd()
        else:
            self.storage = storage
        self.__version__ = "0.0.5"

        if force:
            print('Running.. Saving the Object..\n')
            self._load_light_curve()
            self.save_object(update=True)
        else:
            try:
                self.retrieve_object()
                print('Object was found! \n'
                      'To save manually: save_object(filename, update=True)\n')
            except:
                print('Running.. Saving the Object..\n')
                self._load_light_curve()
                self.save_object(update=True)

    def save_object(self, filename=None, update=False):
        """
        Save the object for later calls
        """
        filename = filename or self.name
        if update:
            file = os.path.join(self.storage, filename)
            with open(file, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(f'Saved at {file}!\n', fg='blue')

    def retrieve_object(self, filename=None):
        filename = filename or self.name
        file = os.path.join(self.storage, filename)
        with open(file, 'rb') as handle:
            click.secho(f'Retrieving object {file}', fg='blue')
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def _load_light_curve(self):
        """ The data are taken from http://asphwww.ph.noda.tus.ac.jp/snn/
        Load data from file;
        File structure is
        `    time_step,\n
             col1 col2 col3 col4 col5 col6 \n
        20x  col1 col2 col3 col4 col5 col6 \n
             col1 col2 col3 col4 col5 col6 \n
        `
        This function is way slower than Ricardo Peres's, but more readable
        """
        mass = self.M
        Z = self.Z
        t_revival = self.t_revival
        available_mass = [13, 20, 30, 50]
        available_Z = [0.02, 0.004]
        available_t_revival = [100, 200, 300]
        assert mass in available_mass, "Required mass not in DB."
        assert Z in available_Z, "Required metallicity not in DB."
        assert t_revival in available_t_revival, "Required revival time not in DB."

        B = 0
        if Z == 0.004: B = 1
        C = int(t_revival / 100)

        try:
            # filepath = f'{paths["data"]}intp{mass}{B}{C}.data'
            filepath = f'../data/rperes_data/intp{mass}{B}{C}.data'
            _f = open(filepath, 'r')
        except:
            filepath = f'/dali/lgrandi/peres/SN/Light_Curve_DB/intpdata/intp{mass}{B}{C}.data'
            _f = open(filepath, 'r')
        f = _f.readlines()
        _f.close()
        # The 0th line is the time
        # following 20 lines are the data
        # there is an empty line and a line for next time step after every 20 lines
        skiprows = [0]
        for i in range(int(len(f) / 22)):
            skiprows.append(21 + i * 22)
            skiprows.append(22 + i * 22)

        # make a file without the empty lines or time lines
        f_filt = [fval for fval in f if f.index(fval) not in skiprows]
        f_new = open('tmp', 'w+')
        f_new.writelines(f_filt)
        f_new.close()
        # remove tmp if you like

        time = np.array([float(fval.split('\n')[0]) for fval in f if f.index(fval) in skiprows and fval != ' \n'])
        E_bins_left, E_bins_right, N_ve, N_ave, N_vx, L_ve, L_ave, L_vx = np.loadtxt('tmp', unpack=True)
        row_nr = int(len(L_vx) / 20)
        # the energies are same at a given time index
        self.t = time  # *u.s
        self.E_bins_left = E_bins_left.reshape(row_nr, 20)[0, :]  # * u.MeV
        self.E_bins_right = E_bins_right.reshape(row_nr, 20)[0, :]  # * u.MeV
        self.N_ve = N_ve.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
        self.N_ave = N_ave.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
        self.N_vx = 4 * N_vx.reshape(row_nr, 20)  # * (u.count*u.MeV**-1 * u.s**-1)
        self.L_ve = L_ve.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
        self.L_ave = L_ave.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
        self.L_vx = L_vx.reshape(row_nr, 20)  # * (u.erg * u.MeV**-1 * u.s**-1)
        self.nu_list = {r'$\nu_e$': self.N_ve,
                        r'$\overline{\nu_e}$': self.N_ave,
                        r'$\sum\nu_x$': self.N_vx}
        self.L_list = {r'L$\nu_e$': self.L_ve,
                       r'L$\overline{\nu_e}$': self.L_ave,
                       r'L$\nu_x$': self.L_vx}
        self.mean_E = (self.E_bins_right + self.E_bins_left) / 2
        return None

    def _get_t_indexes(self, t0=None, tf=None):
        """ find the closest indexes of t0 and tf """
        times = self.t
        t0 = t0 or -np.inf
        tf = tf or np.max(times)
        t0_idx = np.abs(times - t0).argmin()
        tf_idx = np.abs(times - tf).argmin() + 1
        return t0_idx, tf_idx

    def fluxes_at_tpc(self, dist=None, fluxes=None):
        """ Compute the neutrino flux at the detector
            Given a distance to the SN
            Returns counts of (time,energy) size
            (cts/MeV/cm2 , cts/s/cm2)
        """
        dist = dist or self.dist
        dist = ((dist * u.kpc).to(u.cm)).value  # convert distance to cm
        strad = (4 * np.pi * np.power(dist, 2))
        fluxes = fluxes or self.nu_list
        flux_at_tpc = {nu: fluxes[nu] / strad for nu in fluxes.keys()}
        return flux_at_tpc

    def get_flux_at_(self, dist=None, **kwargs):
        """ Returns (interpolated) Neutrino Flux
            if dist is given (in kpc), scales counts by distance
            if 'time' is given, it integrates over energy
                and interpolates time vs fluxes
            if 'energy' is given it integrates over time
                and interpolates energy vs fluxes
            - integrated over all energy bins -
            at a given time t as a dictionary
            e.g. {'nu_e':1e55, 'nu_ae':1e54, 'nu_x':1e57}
        """
        if 'time' in kwargs:
            val = kwargs.get('time')
            flux_for = self.t
            int_over = self.mean_E
            axis = 1
        elif 'energy' in kwargs:
            val = kwargs.get('energy')
            flux_for = self.mean_E
            int_over = self.t
            axis = 0
        else:
            raise ValueError("Options are: time=VAL or energy=VAL")
        if val < int_over.min() or val > int_over.max():
            print(f'{val} is out of bounds ({int_over.min():.1f} {int_over.max():.1f})\n'
                  'The number is extrapolated and might be wrong!')

        # set the fluxes. If a distance is given. Use fluxes at the detector.
        if isinstance(dist, NoneType):
            fluxes = self.nu_list
        else:
            fluxes = self.fluxes_at_tpc(dist)

        # interpolate and calculate the flux
        # at a given value
        nr_at = {nu:
                     itp.interp1d(flux_for, np.trapz(fluxes, int_over, axis=axis),
                                  kind="cubic", fill_value="extrapolate")(val)
                 for nu in self.nu_list.keys()}
        return nr_at

    def _get_t_int_flux(self, fluxes, t0=None, tf=None):
        """ Compute the total neutrino flux integrated over t0-tf
            Returns total count in units of counts/MeV/([tf-t0] sec)
        """
        t0_idx, tf_idx = self._get_t_indexes(t0, tf)
        t_cut = self.t[t0_idx:tf_idx]
        fluxes_cut = fluxes[t0_idx:tf_idx, :]
        integrated_numbers = np.sum(fluxes_cut, axis=0)  # sum or integration?
        # IT NEEDS TO BE SUMMATION.
        # integrated_numbers = np.trapz(fluxes_cut, t_cut, axis=0)
        return integrated_numbers

    def get_integ_fluxes(self, detector=True, t0=None, tf=None, dist=None):
        """ Get integrated number fluxes between t0 and tf
            as a dictionary for each neutrino type
            Returns dictionary with time integrated fluxes
            for each flavour with units (cts/MeV/[tf-t0]sec)
        """
        if detector:
            fluxes = self.fluxes_at_tpc(dist)
        else:
            fluxes = self.nu_list
        t0 = t0 or self.t0
        tf = tf or self.tf
        integ_numbers = {nu: self._get_t_int_flux(fluxes[nu], t0, tf)
                         for nu in fluxes.keys()}
        return integ_numbers

    def _get_rate1D(self, energies, t0, tf, dist):
        """
        Compute scattering rates at the detector at given
        recoil energies for *time integrated* neutrino fluxes. 
        See also _get_rate2D, get_recoil_spectra1D

        Parameters
        ----------
        energies : ndarray
            recoil energies to calculate rates for
        t0,tf : float, optional
            start and end time of the neutrino signal
        dist : float, optional
            supernova distance in units of kpc
        
        Returns
        -------
            Rates as a function of Recoil Energies
        """
        XeNuc = self.XeNuc
        fluxes = self.get_integ_fluxes(detector=True, t0=t0, tf=tf, dist=dist)
        nu_keys = list(fluxes.keys())
        nu_energies = self.mean_E
        flux_interpolator = {nu: itp.interp1d(nu_energies, fluxes[nu],
                                              kind="cubic", fill_value="extrapolate") for nu in nu_keys}
        # For readability
        # Compute rates for each flavor, and for each nuclide
        # convert MeV -> keV, and
        # multiply by the total number of Xe atoms in the TPC
        _nu1 = [xe.dRatedErecoil_vect(energies, Flux=flux_interpolator[nu_keys[0]]) * 1e-3 * N_Xe.value for xe in
                XeNuc]
        _nu2 = [xe.dRatedErecoil_vect(energies, Flux=flux_interpolator[nu_keys[1]]) * 1e-3 * N_Xe.value for xe in
                XeNuc]
        _nu3 = [xe.dRatedErecoil_vect(energies, Flux=flux_interpolator[nu_keys[2]]) * 1e-3 * N_Xe.value for xe in
                XeNuc]

        # sum the rates for each nuclide, 
        nu1 = np.array([np.sum(_nu1, axis=0)])[0]
        nu2 = np.array([np.sum(_nu2, axis=0)])[0]
        nu3 = np.array([np.sum(_nu3, axis=0)])[0]
        # add to self, and compute the total i.e. nu_e + nu_ae + nu_x
        self.rate1D = {nu_keys[0]: nu1, nu_keys[1]: nu2, nu_keys[2]: nu3}
        # call also the total
        self.get_total_rate(dim=1)
        # truncate all rates at 0
        # this updates, rec_en, rate1D, and total_rates1D 
        ## NOTE: self.recoil_en is also updated to have same length!
        self._truncate1D()
        return None

    def get_recoil_spectra1D(self, rec_en=None, t0=None, tf=None, dist=None, _force_calc=0):
        """
        Compute the 1D recoil spectra for all flavors
        Arguments
        ---------
        rec_en   :  array like, optional
            Recoil energies to compute for
            default is 50 energies between 0-20 keV
        t0,tf    :  float, Optional
            Time interval for integrating the fluxes, in sec
            default: 0-10 sec
        dist     :  float, Optional
            SN distance if different than the model
        _force_calc : bool
            If the configuration was run and saved, it is retrieved
            unless it is forced to calculate it again
        Returns
        -------
        None, saves self.rate1D & self.total_rate1D to the object

        See also plot_recoil_spectra()
        """
        t0 = t0 or self.t0
        tf = tf or self.tf
        dist = dist or self.dist
        if rec_en is None:
            recoil_energies = self.recoil_en
        else:
            recoil_energies = rec_en

        ermin, ermax = np.min(recoil_energies), np.max(recoil_energies)
        # update these
        self.recoil_en = recoil_energies
        self.t0 = t0
        self.tf = tf

        # make a name for this run
        name_ = self.name.split('.p')[0]  # fails if different extension is given
        ratename = f'{name_}_Er{ermin:.1f}-{ermax:.1f}_t0-{t0}-tf-{tf}_1D.p'
        if not _force_calc:
            try:  # check if it is saved
                self.retrieve_object(ratename)
                return None
            except:  # if not force to calculate
                return self.get_recoil_spectra1D(rec_en, t0, tf, dist, _force_calc=True)

        print("\nThis will take a minute")
        self._get_rate1D(recoil_energies, t0, tf, dist)
        # overwrite existing object with the updated version
        click.secho(f'Saving {ratename}...', fg='blue')
        self.save_object(ratename, update=True)
        return None

    def _get_rate2D(self, rec_en, dist, step):
        """
        Compute the 2D recoil rate for all flavors
        In both neutrino energies and times.

        Parameters
        ----------
        rec_en : ndarray
            recoil energies to calculate rates for
        dist : float, optional
            supernova distance in units of kpc
        step : int
            step size in time grid. Finer time sampling
            results in long computation times.
        
        Returns
        -------
            2D array with interaction rates in both
            time and energies 
        """
        XeNuc = self.XeNuc
        fluxes = self.fluxes_at_tpc(dist)
        fluxes = {nu: fluxes[nu][::step, :] for nu in fluxes.keys()}
        nu_keys = list(fluxes.keys())
        nu_energies = self.mean_E  # Incident neutrino energy NOT the recoil

        # Make data. i.e. 2D interpolated flux
        Ebins = rec_en
        tbins = self.t[::step]
        # ee, tt = np.meshgrid(Ebins, tbins)

        # make an interpolator, that, at each time step
        # generates an interpolator for neutrino energy <-> flux
        interpolator = {nu:
                        {t: itp.interp1d(nu_energies, fluxes[nu][ti, :], kind="cubic", fill_value="extrapolate")
                         for ti, t in enumerate(tbins)}
                        for nu in nu_keys}

        # rates at each time step, each recoil energy
        # shape is (time, recoil energies)
        _rates_2D = np.zeros(shape=(len(tbins), len(Ebins)))
        rates_2D = {nu: _rates_2D.copy() for nu in nu_keys}
        # # need 1 dictionary with all flavors for each isotope
        # rates_list = [rates_2D] * len(XeNuc)

        # times as vectorized, recoil energies are looped
        for i, _Er in enumerate(tqdm(Ebins)):
            # at each recoil energy, loop over each isotope
            # convert MeV->keV + multiply by Nr of Xe atoms
            raw_rate1 = [xe.dRatedErecoil2D_vect(Er=_Er, Flux=interpolator[nu_keys[0]], t=tbins) * 1e-3 * N_Xe.value for
                         xe in XeNuc]
            raw_rate2 = [xe.dRatedErecoil2D_vect(Er=_Er, Flux=interpolator[nu_keys[1]], t=tbins) * 1e-3 * N_Xe.value for
                         xe in XeNuc]
            raw_rate3 = [xe.dRatedErecoil2D_vect(Er=_Er, Flux=interpolator[nu_keys[2]], t=tbins) * 1e-3 * N_Xe.value for
                         xe in XeNuc]
            # for each isotope the rates are calculated wrt their fraction
            # Thus, multiplying each with N_Xe makes sense.
            rates_2D[nu_keys[0]][:, i] = np.sum(raw_rate1, axis=0)
            rates_2D[nu_keys[1]][:, i] = np.sum(raw_rate2, axis=0)
            rates_2D[nu_keys[2]][:, i] = np.sum(raw_rate3, axis=0)

        self.rates2D = rates_2D
        # get also the total
        self.get_total_rate(dim=2)
        # no trimming for 2D data for the moment
        return rates_2D

    def get_recoil_spectra2D(self, rec_en=None, dist=None, step=1, _force_calc=0):
        """
        Rates will be computed along Neutrino Energies and Time
        Arguments
        ---------
        rec_en   :  array like, optional
            Recoil energies to compute for
            default is 50 energies between 0-20 keV
        dist     :  float, Optional
            SN distance if different than the model
        step     :  int, optional
            To decrease the time steps, a step can be given
        _force_calc : boolean
            If the configuration was run and saved, it is retrieved
            unless it is forced to calculate it again
        Returns
        -------
        None, saves self.rate2D & self.rate_total2D to the object
    
        CAVEAT: This function can take long, 
        as it computes integrals for len(times)*len(Recoil Energies) times 
        """
        dist = dist or self.dist

        if rec_en is None:
            recoil_energies = self.recoil_en
        else:
            recoil_energies = rec_en

        self.recoil_en = recoil_energies  # update
        ermin, ermax = np.min(recoil_energies), np.max(recoil_energies)

        name_ = self.name.split('.p')[0]
        ratename = f'{name_}_Er{ermin:.1f}-{ermax:.1f}_step{step}_dist{dist}_2D.p'
        if not _force_calc:
            try:  # check if it is saved
                self.retrieve_object(ratename)
                return None
            except:
                return self.get_recoil_spectra2D(rec_en, dist, step, _force_calc=True)

        print("\nThis'll take a while")
        self._get_rate2D(recoil_energies, dist, step)
        # overwrite existing object with the updated version
        click.secho(f'Saving {ratename}...', fg='blue')
        self.save_object(ratename, update=True)
        return None

    def get_total_rate(self, dim=1):
        """
        Calculate total rates in 
        all flavors
        For 1D data, `self._truncate1D` is
        called right after this function.
        """
        total = 0
        if dim == 1:
            data = self.rate1D
        elif dim == 2:
            data = self.rates2D
        else:
            raise ValueError
        for name, data_ in data.items():
            total = total + data_
        if dim == 1:
            self.total_rate1D = total
        if dim == 2:
            self.total_rate2D = total
        return None

    def _truncate1D(self):
        """ 
            Once truncated, the self.recoil_en is updated
            Thus, when get_spectra called again, it recalculates
            for these energies.
        """
        flavor_rates = self.rate1D
        total_rates = self.total_rate1D
        rec_en = self.recoil_en
        # truncate all at position where total rate becomes 0
        trunc_here = np.searchsorted(total_rates[::-1], 0, side='right')
        flavor_tr = {nu: flavor_rates[nu][:-trunc_here] for nu in flavor_rates.keys()}
        rates_tr = total_rates[:-trunc_here]
        recen_tr = rec_en[:-trunc_here]
        # update the existing rates
        # if there are no zeros, it returns empty array
        # in this case do not clip any data
        if not len(rates_tr) == 0:
            print('Truncating')
            self.rate1D = flavor_tr
            self.total_rate1D = rates_tr
            self.recoil_en = recen_tr
        return None

    def _get_1Drates_from2D(self, t0=None, tf=None):
        """
        If the 2D rates exists, 1D rates can be 
        computed by integrating them.
        Parameters
        ----------
        t0, tf : float, optional
            start and end time of the flux. If None, uses the limits.
        Returns
        -------
            2 dictionaries;
            1 - for rates for recoil energies integrated along time
            2 - for rates for times integrated along recoil energies
            Both dictionaries having 3 flavours + total rate
        """
        try:
            rates2D = self.rates2D
        except:
            raise ValueError('2D rates do not exist!')
        # flux = self.nu_list
        times = self.t
        rec_energies = self.recoil_en

        # set the time interval
        first_t_idx, last_t_idx = self._get_t_indexes(t0, tf)
        times = times[first_t_idx:last_t_idx]
        rates2D = {nu: rates2D[nu][first_t_idx:last_t_idx, :] for nu in rates2D.keys()}

        N_E, N_t = len(rec_energies), len(times)
        rates_E, rates_t = dict(), dict()

        for nu in rates2D.keys():
            # rates_E[nu] = np.array([np.trapz(rates2D[nu][:,i], times) for i in range(N_E)])
            # rates_t[nu] = np.array([np.trapz(rates2D[nu][i,:], rec_energies) for i in range(N_t)])
            rates_E[nu] = np.sum(rates2D[nu], axis=0)
            rates_t[nu] = np.sum(rates2D[nu], axis=1)

        rates_E['Total'], rates_t['Total'] = 0, 0
        for nu in rates2D.keys():
            rates_E['Total'] += rates_E[nu]
            rates_t['Total'] += rates_t[nu]
        return rates_E, rates_t

    def sample_from_recoil_spectrum(self, x='energy', N_sample=1):
        if x.lower() == 'energy':
            try:
                spectrum_Er, spectrum_t = self._get_1Drates_from2D()
                spectrum = spectrum_Er['Total']
            except:
                # click.secho(f"spectrum does not exist, computing", fg='red')
                # Todo: check here
                spectrum = self.total_rate1D
            xaxis = self.recoil_en
            ## interpolate
            intrp_rates = itp.interp1d(xaxis, spectrum, kind="cubic", fill_value="extrapolate")
            xaxis = np.linspace(xaxis.min(), xaxis.max(), 200)
            spectrum = intrp_rates(xaxis)
        elif x.lower() == 'time':
            # spectrum_Er['Total'] is the same as self.total_rate1D (if run for all time range)
            # it is a seperate if, in case one wants to plot without having 2D data
            spectrum_Er, spectrum_t = self._get_1Drates_from2D()
            spectrum = spectrum_t['Total']
            xaxis = self.t
        else:
            raise KeyError('choose x=time or x=energy')
        sample = _inverse_transform_sampling(xaxis, spectrum, N_sample)
        return sample

    # This allows accessing relative attributes
# i.e. when model = SN_lightcurve(a,b,c) is imported
# model.N_Xe can be called to access N_Xe in constants.py

# import aux_scripts.constants as _constants_attr
# for name in dir(_constants_attr):
#     if not name.startswith('__'):
#         setattr(SN_lightcurve, name, getattr(_constants_attr, name))
