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
import os, click, sys
import numpy as np
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from .Xenon_Atom import ATOM_TABLE
import configparser
from scipy import interpolate
import astropy.units as u
from glob import glob

import snewpy
from snewpy.neutrino import Flavor
from snewpy.models.ccsn import Analytic3Species, Bollig_2016, Fornax_2019, Fornax_2021, Kuroda_2020
from snewpy.models.ccsn import Nakazato_2013, OConnor_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014
from snewpy.models.ccsn import Walk_2018, Walk_2019, Warren_2020, Zha_2021

models_list = ['Analytic3Species', 'Bollig_2016', 'Fornax_2019', 'Fornax_2021', 'Kuroda_2020', 'Nakazato_2013',
               'OConnor_2013', 'OConnor_2015', 'Sukhbold_2015', 'Tamborra_2014', 'Walk_2018', 'Walk_2019',
               'Warren_2020', 'Zha_2021',]

models = [Analytic3Species, Bollig_2016, Fornax_2019, Fornax_2021, Kuroda_2020,
          Nakazato_2013, OConnor_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014,
          Walk_2018, Walk_2019, Warren_2020, Zha_2021]

models_dict = dict(zip(models_list, models))

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

def _parse_models(model_name, filename, index, config):
    """ Get the selected model, or ask user
    """
    try:
        snewpy_base = config['paths']['snewpy_models']
        file_path = os.path.join(snewpy_base, model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0
    except:
        snewpy_base = snewpy.__file__.split("python/snewpy/__init__.py")[0]
        file_path = os.path.join(snewpy_base, "models", model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0
    files_in_model = [f for f in files_in_model if not f.endswith('.md') and not f.endswith('.ipynb')]

    if filename is None:
        _files_in_model_list = [f"[{i}]\t" + f.split("/")[-1] for i, f in enumerate(files_in_model)]
        _files_in_model = "\n".join(_files_in_model_list) + "\n"
        if index is None:
            click.secho("> Available files for this model, please select an index\n\n", fg='blue', bold=True)
            file_index = input(_files_in_model)
        else:
            file_index = index
        selected_file = files_in_model[int(file_index)]
        click.secho(f"> You chose ~wisely~ ->\t   {_files_in_model_list[int(file_index)]}", fg='blue', bold=True)
        return selected_file
    else:
        if filename in [f.split("/")[-1] for i, f in enumerate(files_in_model)]:
            return os.path.join(file_path, filename)
        else:
            raise FileNotFoundError(f"{filename} not found in {file_path}")

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


def add_strax_folder(config):
    """ This appends the SN MC folder to your directories
        So the simulations created by others are accessible to you

    """
    mc_folder = config["wfsim"]["sim_folder"]
    try:
        import strax, cutax
        st = cutax.contexts.xenonnt_sim_SR0v2_cmt_v8(cmt_run_id="026000")
        st.storage += [strax.DataDirectory(os.path.join(mc_folder, "strax_data"), readonly=False)]
    except ImportError:
        pass

class Models:
    """ Deal with a given SN lightcurve from snewpy
    """

    def __init__(self,
                 model_name,
                 filename=None,
                 index=None,
                 model_kwargs=None,
                 distance=10*u.kpc,
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
        # try to find from the default config
        self.config = configparser.ConfigParser()
        self.default_conf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "simple_config.conf")
        conf_path = config_file or self.default_conf_path
        self.config.read(conf_path)

        if model_kwargs is None:
            model_kwargs = dict()
        model_file = _parse_models(model_name, filename, index, config=self.config)
        model = models_dict[model_name](model_file, **model_kwargs)
        self.__dict__.update(model.__dict__)
        self.model = model
        self.composite = composite
        self.Nucleus = get_composite(composite)
        self.distance = distance
        self.recoil_energies = recoil_energies
        self.neutrino_energies = neutrino_energies
        self.name = repr(model).split(":")[1].strip().split('\n')[0]+".pickle"
        self.storage = get_storage(storage, self.config)
        self.fluxes = None
        self.rateper_Er = None
        self.rateper_t = None
        self.__version__ = "1.1.0"
        try:
            self.retrieve_object()
        except FileNotFoundError:
            self.save_object(True)
        add_strax_folder(self.config)

    def __repr__(self):
        """Default representation of the model.
        """
        _repr = self.model.__repr__
        return _repr

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        _repr = self.model._repr_markdown_
        executed = True if self.rateper_Er is not None else False
        s = [_repr, '']
        s += [f"| composite | {self.composite}|"]
        s += [f"| distance | {self.distance}|"]
        s += [f"| executed | {executed}|"]
        return '\n'.join(s)

    def save_object(self, update=False):
        """ Save the object for later calls
        """
        if update:
            file = os.path.join(self.storage, self.name)
            with open(file, 'wb') as output:   # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(f'> Saved at <self.storage>/{self.name}!\n', fg='blue')

    def retrieve_object(self):
        file = os.path.join(self.storage, self.name)
        with open(file, 'rb') as handle:
            click.secho(f'> Retrieving object self.storage/{self.name}', fg='blue')
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def delete_object(self):
        file = os.path.join(self.storage, self.name)
        if input(f"> Are you sure you want to delete\n"
                 f"{file}?\n") == 'y':
            os.remove(file)

    def compute_rates(self, total=True, force=False, leave=False):
        """ Do it for each composite and scale for their abundance
            simple scaling won't work as the proton number changes
            :param total: `bool` if True return total of all isotopes
            :param force: `bool` whether to recalculate if already exist
            :param leave: `bool` tqdm arg, whether to leave the progress bar

            Returns
                (dR/dEr, dR/dt) if total is True, else two dictionaries for both
                containing rates for all isotopes individually
        """
        is_first = True if self.fluxes is None else False
        # create fluxes attribute for each isotope
        # only if fluxes doesn't exist or forced
        for isotope in tqdm(self.Nucleus, total=len(self.Nucleus), desc="Computing for all isotopes", colour="CYAN"):
            isotope.get_fluxes(self.model, self.neutrino_energies, force, leave)
        self.isotope_fluxes = {isotope.name: isotope.fluxes for isotope in self.Nucleus}

        self.rateper_Er_iso = {isotope.name: isotope.dRdEr(self.model, self.neutrino_energies, self.recoil_energies)
                               for isotope in tqdm(self.Nucleus)}
        self.rateper_t_iso = {isotope.name: isotope.dRdt(self.model, self.neutrino_energies, self.recoil_energies)
                              for isotope in tqdm(self.Nucleus)}

        self._compute_total_rates()
        if is_first:
            self.save_object(update=True)
        if total:
            return self.rateper_Er, self.rateper_t
        else:
            return self.rateper_Er_iso, self.rateper_t_iso

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
                     N_Xe=4.6e27*u.count/u.tonne,
                     distance=10*u.kpc,
                     overwrite=False):
        """ Scale fluxes based on distance and number of atoms
            Return: scaled fluxes
        """
        scale = N_Xe / (4 * np.pi * distance ** 2).to(u.m ** 2)
        try:
            fluxes_scaled = {}
            for f in self.fluxes.keys():
                if overwrite:
                    self.fluxes[f] *= scale
                    return self.fluxes
                else:
                    fluxes_scaled[f] = self.fluxes[f] * scale
                    return fluxes_scaled
        except:
            raise NotImplementedError("fluxes does not exist\nCreate them by calling `get_fluxes()`")

    def scale_rates(self,
                    isotopes=False,
                    N_Xe=4.6e27*u.count/u.tonne,
                    distance=10*u.kpc,
                    overwrite=False):
        """ Scale rates based on distance, and number of atoms
            Return: scaled rates (rates_Er, rates_t)
        """
        scale = N_Xe / (4 * np.pi * distance ** 2).to(u.m ** 2)
        try:
            if isotopes:
                if overwrite:
                    for f in self.rateper_Er_iso.keys():
                        self.rateper_Er_iso[f] *= scale
                        self.rateper_t_iso[f] *= scale
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

    def _inverse_transform_sampling(self, x_vals, y_vals, n_samples):
        cum_values = np.zeros(x_vals.shape)
        y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
        cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
        inv_cdf = interpolate.interp1d(cum_values / np.max(cum_values), x_vals)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    def sample_data(self, n, dtype='energy', return_xy=False):
        if dtype=="energy":
            xaxis = self.recoil_energies
            yaxis = self.rateper_Er['Total']
        else:
            xaxis = self.time
            yaxis = self.rateper_t['Total']

        data = self._inverse_transform_sampling(xaxis, yaxis, n)
        if return_xy:
            return data, xaxis, yaxis
        else:
            return data

    def simulate_one(self, df, runid, context=None, config=None):
        config = config or self.config
        from .Simulate import _simulate_one
        return _simulate_one(df, runid, config=config, context=context)