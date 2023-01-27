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

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

from .snewpy_models import SnewpyWrapper, models_list
import configparser
import astropy.units as u
from snewpy.neutrino import Flavor
from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

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

class Models:
    """ Deal with a given SN lightcurve from snewpy
    """

    def __init__(self,
                 model_name,
                 storage=None,
                 config_file=None,
                 ):
        """
        Parameters
        ----------
        :param model_name: `str`, name of the model e.g. "Nakazato_2013"
        :param storage: `str` path of the output folder
        :param config_file: `str` config file that contains the default params
        """
        self.user = os.environ['USER']
        self.config = configparser.ConfigParser()
        self.default_conf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "simple_config.conf")
        conf_path = config_file or self.default_conf_path
        self.config.read(conf_path)
        self.model_name = model_name
        self.model_caller = SnewpyWrapper(self.model_name, self.config)
        self.model = None
        self.storage = get_storage(storage, self.config)
        self.object_name = ""
        # parameters to use for computations
        self.recoil_energies = np.linspace(0, 20, 100) * u.keV
        self.neutrino_energies = np.linspace(0, 200, 100)*u.MeV
        self.time_range = (None, None)
        self.times = None
        # computed attributes
        self.fluxes = None
        print(f"> {self.model_name} is created, load a progenitor by function call.")

    def __call__(self, filename=None, index=None, force=False, savename=None, **model_kwargs):
        # set the default name

        if savename is None:
            self.model = self.model_caller.load_model_data(filename, index, force)
            savename = ("-".join(self.model_caller.selected_file.split("/")[-2:])).replace('.', '_') + ".pickle"

        # check if file exists
        full_file_path = os.path.join(self.storage, savename)
        if os.path.isfile(full_file_path):
            # try to retrieve
            self.retrieve_object(savename)
        else:
            # create that object and save
            # self.model = self.model_caller.load_model_data(filename, index, force)
            self.object_name = savename
            self.times = self.model.time
            self.time_range = (self.times[0], self.times[-1])
            self.save_object(update=True)

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

        model_loaded = True if self.model is not None else False
        s = [_repr]
        if model_loaded:
            s += [f"|file name| {self.object_name}"]
            s += [f"|duration | {np.round(np.ptp(self.model.time), 2)}|"]
            s += [f"|time range| {self.time_range}"]
        return '\n'.join(s)


    def set_params(self, time_samples=None, neutrino_energies=None):
        """ Set the parameters for calculations
            If the fluxes have already been computed this will ask for a confirmation

            :param neutrino_energies: `array` default np.linspace(0,200,100)*u.MeV
                if provided without units, assumes MeV
            :param time_samples: `array` if given, time_range is ignored, default = model times
                if no unit is given, assumes seconds

        """
        neutrino_energies = neutrino_energies or self.recoil_energies
        time_samples = time_samples or self.times

        if not type(neutrino_energies) == u.quantity.Quantity:
            # assume MeV
            neutrino_energies *= u.MeV

        if not type(time_samples) == u.quantity.Quantity:
            # assume seconds
            time_samples *= u.s

        # if the user has changed something the rates needs to be recomputed. Ask for confirmation
        if not all([neutrino_energies == self.neutrino_energies,
                    time_samples == self.times]) and self.fluxes is not None:
            self.fluxes = None
        self.neutrino_energies = neutrino_energies
        self.times = time_samples


    def save_object(self, update=False):
        """ Save the object for later calls
        """
        if update:
            full_file_path = os.path.join(self.storage, self.object_name)
            with open(full_file_path, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(f'> Saved at <self.storage>/{self.object_name}!\n', fg='blue')
            # self.history = make_history(self.history, "Data Saved!", self.__version__, self.user)

    def retrieve_object(self, name=None):
        file = name or self.object_name
        full_file_path = os.path.join(self.storage, file)
        with open(full_file_path, 'rb') as handle:
            print(f">>>>> {file}")
            click.secho(f'> Retrieving object self.storage{file}', fg='blue')
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def delete_object(self):
        full_file_path = os.path.join(self.storage, self.object_name)
        if input(f"> Are you sure you want to delete\n"
                 f"{full_file_path}?\n") == 'y':
            os.remove(full_file_path)

    def compute_model_fluxes(self,
                             time_samples=None,
                             neutrino_energies=None,
                             force=False,
                             leave=False, **kw):
        """ Compute fluxes for each time and neutrino combination for each neutrino flavor
            Result is a dictionary saved in self.fluxes with keys representing each flavor,
            and each key having MxN matrix for fluxes
        :param neutrino_energies: `array` default np.linspace(0, 200, 100) * u.MeV
            if provided without units, assumes MeV
        :param time_samples: `array` if given, time_range is ignored, default = model_times
            if no unit is given, assumes seconds

        if you are forcing, you might want to change the name first, so it doesn't overwrite
        """
        if self.fluxes is not None and not force:
            # already computed
            return None

        # sets the object attributes, and resets the fluxes if things have changed
        self.set_params(time_samples=time_samples, neutrino_energies=neutrino_energies)

        # get fluxes at each time and at each neutrino energy
        flux_unit = self.model.get_initial_spectra(1*u.s, 100*u.MeV, **kw)[Flavor.NU_E].unit

        # create a flux dictionary for each flavor
        _fluxes = np.zeros((len(self.times), len(self.neutrino_energies))) * flux_unit
        _fluxes = {f: _fluxes.copy() for f in Flavor}

        for f in tqdm(Flavor, total=len(Flavor), desc="Computing Fluxes", leave=leave):
            for i, sec in tqdm(enumerate(self.times), total=len(self.times), desc=f.to_tex(), leave=False):
                _fluxes_dict = self.model.get_initial_spectra(sec, self.neutrino_energies, **kw)
                _fluxes[f][i, :] = _fluxes_dict[f]
        self.fluxes = _fluxes
        self.save_object(update=True)


    def scale_fluxes(self, distance, N_Xe=4.6e27*u.count/u.tonne, overwrite=False):
        """ Scale fluxes based on distance and number of atoms
            Return: scaled fluxes
        """
        if not type(distance) == u.quantity.Quantity:
            # assume kpc
            distance *= u.kpc

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
