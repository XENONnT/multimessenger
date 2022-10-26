#!/usr/bin/python
"""
Last Update: 17-08-2022
------------------------d
Supernova Models module.
Methods to deal with supernova lightcurve and derived properties

Author: Melih Kara kara@kit.edu

Notes
-----
uses _pickle module, check here https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
How to pickle yourself https://stackoverflow.com/questions/2709800/how-to-pickle-yourself

"""
import os, click
import numpy as np
import pandas as pd

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from .Xenon_Atom import ATOM_TABLE
from .sn_utils import _inverse_transform_sampling
from .snewpy_models import fetch_model_name, fetch_model
from .snewpy_models import models_list
import configparser
import astropy.units as u
from datetime import datetime
from snewpy.neutrino import Flavor

from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def get_composite(composite):
    """ Get a Xenon nucleus composite
    """
    from .Recoil_calculations import TARGET
    if composite == "Xenon":
        Nucleus = [TARGET(ATOM_TABLE["Xe124"], pure=False),
                   TARGET(ATOM_TABLE["Xe126"], pure=False),
                   TARGET(ATOM_TABLE["Xe128"], pure=False),
                   TARGET(ATOM_TABLE["Xe129"], pure=False),
                   TARGET(ATOM_TABLE["Xe130"], pure=False),
                   TARGET(ATOM_TABLE["Xe131"], pure=False),
                   TARGET(ATOM_TABLE["Xe132"], pure=False),
                   TARGET(ATOM_TABLE["Xe134"], pure=False),
                   TARGET(ATOM_TABLE["Xe136"], pure=False)]
    else:
        raise NotImplementedError(f"{composite} Requested but only 'Xenon' is implemented so far")
    return Nucleus

def get_storage(storage, config):
    if storage is None:
        # where the snewpy models saved, ideally we want a single place
        try:
            storage = config['paths']['processed_data']
            os.path.isdir(storage) # check if we are on dali
        except KeyError as e:
            print(f"> KeyError: {e} \nSetting current directory as the storage, "
                  f"pass storage=<path-to-your-storage> ")
            storage = os.getcwd()
    else:
        storage = storage
    return storage

def add_strax_folder(config, context=None):
    """ This appends the SN MC folder to your directories
        So the simulations created by others are accessible to you

    """
    mc_folder = config["wfsim"]["sim_folder"]
    mc_data_folder = os.path.join(mc_folder, "strax_data")
    try:
        import strax, cutax
        st = context or cutax.contexts.xenonnt_sim_SR0v2_cmt_v8(cmt_run_id="026000",
                                                                output_folder=mc_data_folder)
        output_folder_exists = False
        for i, stores in enumerate(st.storage):
            if mc_data_folder in stores.path:
                output_folder_exists = True
        if not output_folder_exists:
            st.storage += [strax.DataDirectory(mc_data_folder, readonly=False)]
        click.secho(f"Using 'cutax.contexts.xenonnt_sim_SR0v2_cmt_v8' \nsaving data to {mc_data_folder}", fg='blue')
        return st
    except ImportError:
        click.secho("> You don't have strax/cutax, won't be able to simulate!", fg='red')
        pass

def make_history(history, input_str, version, user=None, fmt='%Y/%m/%d - %H:%M UTC'):
    user = user or ""
    now = datetime.utcnow().strftime(fmt)
    new_df = pd.DataFrame({"date":now, "version":version, "user":user, "history":input_str},
    columns=['date', 'version', 'user', 'history'], index=[0])
    history = pd.concat([history, new_df], ignore_index=True)
    return history


class Models:
    """ Deal with a given SN lightcurve from snewpy
    """

    def __init__(self,
                 model_name,
                 filename=None,
                 index=None,
                 model_kwargs=None,
                 distance=10*u.kpc,
                 volume=5.9*u.tonne,
                 recoil_energies=np.linspace(0,20,100)*u.keV,
                 neutrino_energies=np.linspace(0, 200, 100)*u.MeV,
                 composite="Xenon",
                 storage=None,
                 config_file=None,
                 ):
        """
        Parameters
        ----------
        :param model_name: `str`, name of the model e.g. "Nakazato_2013"
        :param filename: `str` name of the file e.g. "nakazato-shen-z0.02-t_rev100ms-s50.0.fits"
        :param index: `int` file index, if known
        :param model_kwargs: `dict` any model-specific parameters
        :param distance: `float` Supernova distance, default is 10 kpc
        :param recoil_energies: `array` recoil energies to search for interactions
        :param neutrino_energies: `array` neutrino energy sampling
        :param composite: `str` What nucleus to use ("Xenon" only for now)
        :param storage: `str` path of the output folder
        :param config_file: `str` config file that contains the default params
        """
        self.user = os.environ['USER']
        self.model_name = model_name
        # try to find from the default config
        self.config = configparser.ConfigParser()
        self.default_conf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "simple_config.conf")
        conf_path = config_file or self.default_conf_path
        self.config.read(conf_path)
        self.model_kwargs = model_kwargs or dict()
        self.model_input = dict(model_name=self.model_name, filename=filename, index=index, config=self.config)
        self.model_file = fetch_model_name(**self.model_input)
        self.composite = composite
        self.N_Xe = 4.6e27 * u.count / u.tonne
        self.Nucleus = get_composite(composite)
        self.distance = distance
        self.volume = volume
        self.recoil_energies = recoil_energies
        self.neutrino_energies = neutrino_energies
        self.name = ("-".join(self.model_file.split("/")[-2:])).replace('.', '_')+".pickle"
        if self.model_name == "Warren_2020":
            self.name = "Warren_2020_"+self.name
        self.storage = get_storage(storage, self.config)
        self.fluxes = None
        self.rateper_Er = None
        self.rateper_t = None
        self.single_rate = None
        self.history = pd.DataFrame(columns=['date', 'version', 'user', 'history'])
        self.simulation_history = {}
        self.__version__ = "1.2.1"
        try:
            self.retrieve_object()
        except FileNotFoundError:
            self.model = fetch_model(self.model_name, self.model_file, **self.model_kwargs)
            self.save_object(True)



    def __repr__(self):
        """Default representation of the model.
        """
        _repr = self.model.__repr__()
        return _repr

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        try:
            _repr = self.model._repr_markdown_()
        except AttributeError:
            _repr = f"**{self.model_name}**"
        executed = True if self.rateper_Er is not None else False
        s = [_repr]
        s += [f"|composite | {self.composite}|"]
        s += [f"|duration | {np.round(np.ptp(self.model.time), 2)}|"]
        s += [f"|distance | {self.distance}|"]
        s += [f"|volume | {self.volume}|"]
        s += [f"|executed | {executed}|"]
        if executed:
            s += [f"|SN rate | {int(self.single_rate.value)} ct|"]
        return '\n'.join(s)

    def save_object(self, update=False):
        """ Save the object for later calls
        """
        if self.model_name in ["Fornax_2019","OConnor_2013"]:
            self._handle_h5files('save', update=update)
            return None
        if update:
            file = os.path.join(self.storage, self.name)
            with open(file, 'wb') as output:   # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(f'> Saved at <self.storage>/{self.name}!\n', fg='blue')
            self.history = make_history(self.history, "Data Saved!", self.__version__, self.user)

    def retrieve_object(self):
        if self.model_name in ["Fornax_2019","OConnor_2013"]:
            self._handle_h5files('retrieve')
            return None
        file = os.path.join(self.storage, self.name)
        with open(file, 'rb') as handle:
            click.secho(f'> Retrieving object self.storage/{self.name}', fg='blue')
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def _handle_h5files(self, mode, update=False):
        """ Fornax 2019, has hdf5 file which cannot be pickled
        Thu, I remove the model attr first and store, and while retrieving
        I append the model attr back
        """
        file = os.path.join(self.storage, self.name)
        if mode == 'save':
            if update:
                self.model = None
                with open(file, 'wb') as output:  # Overwrites any existing file.
                    pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                    click.secho(f'> Saved at <self.storage>/{self.name}!\n', fg='blue')
                self.history = make_history(self.history, "Data Saved!", self.__version__, self.user)
        elif mode == 'retrieve':
            with open(file, 'rb') as handle:
                click.secho(f'> Retrieving object self.storage/{self.name}', fg='blue')
                tmp_dict = pickle.load(handle)
            self.__dict__.update(tmp_dict.__dict__)
            self.model = fetch_model(self.model_name, self.model_file, **self.model_kwargs)
            # self.__dict__.update(self.model.__dict__)
            return None
        else:
            raise FileNotFoundError(f"mode={mode} passed, but what even is it?")

    def delete_object(self):
        file = os.path.join(self.storage, self.name)
        if input(f"> Are you sure you want to delete\n"
                 f"{file}?\n") == 'y':
            os.remove(file)

    def compute_rates(self, total=True, force=False, leave=False, return_vals=False, **kw):
        """ Do it for each composite and scale for their abundance
            simple scaling won't work as the proton number changes
            :param total: `bool` if True return total of all isotopes
            :param force: `bool` whether to recalculate if already exist
            :param leave: `bool` tqdm arg, whether to leave the progress bar
            :param return_vals: `bool` whether to return rates or just create attributes

            Returns
                (dR/dEr, dR/dt) if total is True, else two dictionaries for both
                containing rates for all isotopes individually
        """
        is_first = True if self.fluxes is None else False
        # create fluxes attribute for each isotope
        # only if fluxes doesn't exist or forced
        for isotope in tqdm(self.Nucleus, total=len(self.Nucleus), desc="Computing for all isotopes", colour="CYAN"):
            isotope.get_fluxes(self.model, self.neutrino_energies, force, leave, **kw)
        self.isotope_fluxes = {isotope.name: isotope.fluxes for isotope in self.Nucleus}
        self.rateper_Er_iso = {isotope.name: isotope.dRdEr(self.model, self.neutrino_energies, self.recoil_energies)
                               for isotope in tqdm(self.Nucleus)}
        self.rateper_t_iso = {isotope.name: isotope.dRdt(self.model, self.neutrino_energies, self.recoil_energies)
                              for isotope in tqdm(self.Nucleus)}

        self._compute_total_rates()
        if is_first:
            self.history = make_history(self.history, "Fluxes computed!", self.__version__, self.user)
            self.save_object(update=True)

        _str = "(use scale_rates() for distance & volume)"
        if return_vals:
            if total:
                click.secho(f"> Returning the -total- rates at the source for 1 atom {_str}", fg='blue')
                return self.rateper_Er, self.rateper_t
            else:
                click.secho(f"> Returning the rates -per isotope- at the source for 1 atom {_str}", fg='blue')
                return self.rateper_Er_iso, self.rateper_t_iso
        click.secho(f"> Rates are computed at the source for 1 atom see rateper_Er/t attr {_str}", fg='green')

    def _compute_total_rates(self):
        # get the total fluxes and rates
        f_example = self.isotope_fluxes[self.Nucleus[0].name][Flavor.NU_E]
        dEr_example = self.rateper_Er_iso[self.Nucleus[0].name][Flavor.NU_E]
        dt_example = self.rateper_t_iso[self.Nucleus[0].name][Flavor.NU_E]

        self.fluxes = {f: np.zeros_like(f_example) for f in Flavor}
        self.rateper_Er = {f: np.zeros_like(dEr_example) for f in Flavor}
        self.rateper_t = {f: np.zeros_like(dt_example) for f in Flavor}
        for f in Flavor:
            for xe in self.Nucleus:
                self.fluxes[f] += self.isotope_fluxes[xe.name][f]
                self.rateper_Er[f] += self.rateper_Er_iso[xe.name][f]
                self.rateper_t[f] += self.rateper_t_iso[xe.name][f]
        # get Total in all flavors
        self.rateper_Er["Total"], self.rateper_t["Total"] = 0, 0
        for f in Flavor:
            self.rateper_Er["Total"] += self.rateper_Er[f]
            self.rateper_t["Total"] += self.rateper_t[f]
        # leave out the totals for individual isotopes for later

        # once the total rates computed, also store the expected event rate at default distance
        scaled_total_rate_Er, scaled_total_rate_t = self.scale_rates(distance=self.distance)
        self.single_rate = np.trapz(scaled_total_rate_Er['Total'] * self.volume, self.recoil_energies)

    def truncate_rates(self):
        """ Truncate rates and recoil energies and times
        Returns: rates_Er_tr, rates_t_tr, recen_tr, times_tr
        """
        recoil_energies = self.recoil_energies
        rates_Er = self.rateper_Er
        times = self.model.time
        rates_t = self.rateper_t

        # truncate all at position where total rate becomes 0
        # rateper_Er
        trunc_here = np.searchsorted(rates_Er['Total'][::-1], 0, side='right')
        rates_Er_tr = {nu: rates_Er[nu][:-trunc_here] for nu in rates_Er.keys()}
        recen_tr = recoil_energies[:-trunc_here]
        # rateper_t
        trunc_here = np.searchsorted(rates_t['Total'][::-1], 0, side='right')
        rates_t_tr = {nu: rates_t[nu][:-trunc_here] for nu in rates_t.keys()}
        times_tr = times[:-trunc_here]
        return rates_Er_tr, rates_t_tr, recen_tr, times_tr

    def scale_fluxes(self,
                     distance=None,
                     overwrite=False):
        """ Scale fluxes based on distance and number of atoms
            Return: scaled fluxes
        """
        distance = distance or self.distance
        scale = self.N_Xe / (4 * np.pi * distance ** 2).to(u.m ** 2)
        try:
            fluxes_scaled = {}
            for f in self.fluxes.keys():
                if overwrite:
                    self.fluxes[f] *= scale
                    self.history = make_history(self.history, "Scaled fluxes overwritten the self.fluxes!",
                                                self.__version__, self.user)
                    return self.fluxes
                else:
                    fluxes_scaled[f] = self.fluxes[f] * scale
                    return fluxes_scaled
        except:
            raise NotImplementedError("fluxes does not exist\nCreate them by calling `get_fluxes()`")

    def scale_rates(self,
                    isotopes=False,
                    distance=None,
                    overwrite=False):
        """ Scale rates based on distance, and number of atoms
            :param isotopes: `bool` whether to scale isotope rates or total
            :param distance: `float*unit` default is self.distance
            :param overwrite: `bool` if True overwrites the default attributes i.e. rates at the source
            Return: scaled rates (rates_Er, rates_t)
        """
        distance = distance or self.distance
        scale = self.N_Xe / (4 * np.pi * distance ** 2).to(u.m ** 2)
        try:
            if isotopes:
                if overwrite:
                    for f in self.rateper_Er_iso.keys():
                        self.rateper_Er_iso[f] *= scale
                        self.rateper_t_iso[f] *= scale
                    self.history = make_history(self.history, "Scaled rates overwritten the self.rateper_Er(t)_iso!",
                                                self.__version__, self.user)
                    return self.rateper_Er_iso, self.rateper_t_iso
                else:
                    rates_Er_iso_scaled = {}
                    rates_t_iso_scaled = {}
                    for f in self.rateper_Er_iso.keys():
                        rates_Er_iso_scaled[f] = self.rateper_Er_iso[f] * scale
                        rates_t_iso_scaled[f] = self.rateper_t_iso[f] * scale
                    return rates_Er_iso_scaled, rates_t_iso_scaled
            else:
                if overwrite:
                    for f in self.rateper_Er.keys():
                        self.rateper_Er[f] *= scale
                        self.rateper_t[f] *= scale
                    self.history = make_history(self.history,"Scaled rates overwritten the self.rateper_Er(t)!",
                                                self.__version__, self.user)
                    return self.rateper_Er, self.rateper_t
                else:
                    rates_Er_scaled = {}
                    rates_t_scaled = {}
                    for f in self.rateper_Er.keys():
                        rates_Er_scaled[f] = self.rateper_Er[f] * scale
                        rates_t_scaled[f] = self.rateper_t[f] * scale
                    return rates_Er_scaled, rates_t_scaled
        except:
            raise NotImplementedError("Rates does not exist\nCreate them by calling `compute_rates()`")

    def sample_data(self, n, dtype='energy', return_xy=False):
        if dtype=="energy":
            xaxis = self.recoil_energies
            yaxis = self.rateper_Er['Total']
        else:
            xaxis = self.model.time
            yaxis = self.rateper_t['Total']

        data = _inverse_transform_sampling(xaxis, yaxis, n)
        if return_xy:
            return data, xaxis, yaxis
        else:
            return data

    def generate_instructions(self, energy_deposition,
                              n_tot=1000,
                              rate=20.,
                              fmap=None, field=None,
                              nc=None,
                              r_range=(0, 66.4), z_range=(-148.15, 0),
                              mode="all",
                              timemode="realistic",
                              time_offset=0,
                              distance=None,
                              volume=None
                              ):
        volume = volume or self.volume
        distance = distance or self.distance
        _locals = locals()
        _locals.pop("distance")
        _locals.pop("volume")
        # need the rates in single SN for shifted time sampling
        scaled_total_rate_Er, scaled_total_rate_t = self.scale_rates(distance=distance)
        # total interactions expected from single SN
        _single_rate = np.trapz(scaled_total_rate_Er['Total'] * volume, self.recoil_energies)
        kwargs = dict(recoil_energies=self.recoil_energies,
                      times=self.model.time,
                      rates_per_Er=self.rateper_Er['Total'],
                      rates_per_t=self.rateper_t['Total'],
                      total=n_tot,
                      rate_in_oneSN=_single_rate)

        from .Simulate import generate_sn_instructions
        return generate_sn_instructions(**_locals, **kwargs)


    def simulate_one(self, df, runid, context=None, config=None):
        self.history = make_history(self.history, f"simulation {runid} is requested!", self.__version__, self.user)
        config = config or self.config
        _context = add_strax_folder(config, context)
        from .Simulate import _simulate_one
        self._make_simulation_history(_context, runid, len(df))
        return _simulate_one(df, runid, config=config, context=_context)


    def _make_simulation_history(self,  st, runid, size, fmt='%Y/%m/%d - %H:%M UTC'):
        import strax, cutax, straxen, wfsim
        _vers = f"strax:{strax.__version__} straxen:{straxen.__version__} cutax:{cutax.__version__} wfsim:{wfsim.__version__}"

        cols = ['context hash', 'runid', 'date', 'user', 'model', 'single SN events', 'size']
        curr_sim = self.simulation_history.get(_vers, pd.DataFrame(columns=cols))

        new_dict = {'context hash': st._context_hash(),
                    'runid': runid,
                    'date': datetime.utcnow().strftime(fmt),
                    'user': self.user,
                    'model': self.name,
                    'single SN events':int(self.single_rate.value),
                    'size': size,}
        new_df = pd.DataFrame(new_dict, columns=cols, index=[0])
        curr_sim = pd.concat([curr_sim, new_df], ignore_index=True)
        self.simulation_history[_vers] = curr_sim
        self.save_object(update=True)

    @property
    def display_simulation_history(self):
        if len(self.simulation_history)==0:
            return pd.DataFrame()
        versions, data = list(self.simulation_history.items())[0]
        df = pd.concat([data], keys=[versions], names=('versions', 'index'))
        return df

    @property
    def display_history(self):
        return self.history.copy()
