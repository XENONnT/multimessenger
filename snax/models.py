#!/usr/bin/python
"""
Last Update: 13-02-2024
-----------------------
Supernova Models module.
Methods to deal with supernova lightcurve and derived properties

Author: Melih Kara kara@kit.edu

Notes
-----
uses _pickle module, check here https://stackoverflow.com/questions/4529815/saving-an-object-data-persistence
How to pickle yourself https://stackoverflow.com/questions/2709800/how-to-pickle-yourself

"""
import click
import os
import numpy as np

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

import configparser
import astropy.units as u
from snewpy.neutrino import Flavor
from .sn_utils import isnotebook, deterministic_hash, validate_config_file, get_hash_from_model
import copy

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class SnaxModel:
    """Build on top of snewpy model use either an external snewpy model
    or a SnewpyModel object to get the models
    """

    def __init__(self, snewpy_model, config_file=None):
        """
        Parameters
        ----------
        :param snewpy_model: snewpy model object, either from snewpy or from SnewpyModel
        :param config_file: `str`, optional path to config file that contains the default params

        example:
        --------
        from snewpy.models.ccsn import Nakazato_2013
        nakazato = Nakazato_2013(progenitor_mass=20*u.solMass, revival_time=100*u.ms, metallicity=0.004, eos='shen')
        snax = SnaxModel(nakazato)
        --- or --- RECOMMENDED IF YOU ARE ON THE CLUSTER
        snewpy_model = SnewpyModel()
        nakazato = snewpy_model('Nakazato_2013', combination_index=1)
        model = SnaxModel(nakazato)
        """
        # get the proc_loc
        self._get_config(config_file)
        # model related attributes
        self.model = snewpy_model
        self.model_name = self.model.__class__.__name__
        self.times = snewpy_model.time
        # parameters to use for computations
        self.neutrino_energies = np.linspace(0, 200, 100) * u.MeV
        self.time_range = (None, None)
        # computed attributes
        self.fluxes = None
        self.scaled_fluxes = None
        # find a deterministic hash for the model
        self.snewpy_hash = get_hash_from_model(snewpy_model, snewpy_model.__name__)
        self.model_hash = self._find_hash() # allow for different neutrino energy range and time range
        self.object_name = f"sn_{self.model_name}_{self.model_hash}.pkl"
        # retrieve object if exists
        self.__call__()

    def _get_config(self, config_file):
        """Get the config file and the proc_loc
        if not provided, use the default config file
        """
        self.user = os.environ["USER"]
        default_base_midway_path = "/project2/lgrandi/xenonnt/simulations/supernova"
        if os.path.exists(default_base_midway_path):
            # use the default config file
            conf_path = os.path.join(
                os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
            )
        else:
            # use the local config file
            conf_path = "temp_config.ini"

        # test user input
        if config_file is not None and os.path.exists(config_file):
            is_valid, _ = validate_config_file(config_file)
            if is_valid:
                conf_path = config_file
        self.config_path = conf_path
        # read the config file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        # this is where we store the processed data
        self.proc_loc = self.config["paths"]["processed_data"]

    def _find_hash(self):
        """Find a deterministic hash for the model"""
        # get the parameters
        _meta = {"snewpy_hash": self.snewpy_hash, "nu_energy": self.neutrino_energies.value,
                 "time_range": self.time_range}
        # get the hash
        return deterministic_hash(_meta)

    def __call__(self, force=False):
        """Call the model and save the output
        :param force: `bool` if True, will overwrite the existing file
        """
        # check if file exists
        full_file_path = os.path.join(self.proc_loc, self.object_name)
        if os.path.isfile(full_file_path) and not force:
            # try to retrieve
            self.retrieve_object(self.object_name)
        else:
            # create that object and save
            self.object_name = self.object_name
            self.times = self.model.time
            self.time_range = (self.times[0], self.times[-1])
            self.save_object(update=True)

    def __repr__(self):
        """Default representation of the model."""
        _repr = self.model.__repr__()
        return _repr

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks."""
        try:
            _repr = self.model._repr_markdown_()
        except AttributeError:
            _repr = f"**{self.model_name}**"

        model_loaded = True if self.model is not None else False
        s = [_repr]
        if model_loaded:
            s += [f"|file name| {self.object_name}"]
            s += [f"|duration | {np.round(np.ptp(self.model.time), 2)}|"]
            s += [f"|time range| ({self.time_range[0]}, {self.time_range[1]})"]
        return "\n".join(s)

    def set_params(self, time_samples=None, neutrino_energies=None):
        """Set the parameters for calculations
        If the fluxes have already been computed this will ask for a confirmation

        :param neutrino_energies: `array` default np.linspace(0,200,100)*u.MeV
            if provided without units, assumes MeV
        :param time_samples: `array` if given, time_range is ignored, default = model times
            if no unit is given, assumes seconds

        """
        neutrino_energies = (
            neutrino_energies
            if type(neutrino_energies) != type(None)
            else self.neutrino_energies
        )
        time_samples = time_samples if type(time_samples) != type(None) else self.times

        if not type(neutrino_energies) == u.quantity.Quantity:
            # assume MeV
            neutrino_energies *= u.MeV

        if not type(time_samples) == u.quantity.Quantity:
            # assume seconds
            time_samples *= u.s

        # if the user has changed something the rates needs to be recomputed
        if (
            not (
                np.array_equal(neutrino_energies, self.neutrino_energies)
                and np.array_equal(time_samples, self.times)
            )
            and self.fluxes is not None
        ):
            self.fluxes = None

        # set those params
        self.neutrino_energies = neutrino_energies
        self.times = time_samples

    def save_object(self, update=False):
        """Save the object for later calls"""
        if update:
            full_file_path = os.path.join(self.proc_loc, self.object_name)
            with open(full_file_path, "wb") as output:  # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(
                    f"> Saved at <self.proc_loc>/{self.object_name}!\n", fg="blue"
                )

    def retrieve_object(self, name=None):
        file = name or self.object_name
        full_file_path = os.path.join(self.proc_loc, file)
        with open(full_file_path, "rb") as handle:
            print(f">>>>> {file}")
            click.secho(f"> Retrieving object self.proc_loc{file}", fg="blue")
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def delete_object(self):
        full_file_path = os.path.join(self.proc_loc, self.object_name)
        if input(f"> Are you sure you want to delete\n" f"{full_file_path}?\n") == "y":
            os.remove(full_file_path)

    def compute_model_fluxes(
        self, time_samples=None, neutrino_energies=None, force=False, leave=False, **kw
    ):
        """Compute fluxes for each time and neutrino combination for each neutrino flavor
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
            click.secho(
                "Fluxes already exist in `self.fluxes`, and force=False, doing nothing."
            )
            return None

        # sets the object attributes, and resets the fluxes if things have changed
        self.set_params(time_samples=time_samples, neutrino_energies=neutrino_energies)

        # get fluxes at each time and at each neutrino energy
        flux_unit = self.model.get_initial_spectra(1 * u.s, 100 * u.MeV, **kw)[
            Flavor.NU_E
        ].unit

        # create a flux dictionary for each flavor
        _fluxes = np.zeros((len(self.times), len(self.neutrino_energies))) * flux_unit
        _fluxes = {f: _fluxes.copy() for f in Flavor}

        for i, sec in tqdm(
            enumerate(self.times), total=len(self.times), desc="Looping", leave=leave
        ):
            _fluxes_dict = self.model.get_initial_spectra(
                sec, self.neutrino_energies, **kw
            )
            # for f in tqdm(Flavor, total=len(Flavor), leave=False):
            for f in Flavor:
                _fluxes[f][i, :] = _fluxes_dict[f]
        self.fluxes = _fluxes
        self.save_object(update=True)

    def scale_fluxes(self, distance, N_Xe=4.6e27 * u.count / u.tonne):
        """Scale fluxes based on distance and number of atoms
        distance is assumed to be given in kpc or with units

        Return: scaled fluxes
        """
        # each time copy from the fluxes
        self.scaled_fluxes = copy.deepcopy(self.fluxes)
        if not type(distance) == u.quantity.Quantity:
            # assume kpc
            distance *= u.kpc

        scale = N_Xe / (4 * np.pi * distance**2).to(u.m**2)
        try:
            for f in self.scaled_fluxes.keys():
                self.scaled_fluxes[f] *= scale
            return self.scaled_fluxes
        except Exception as e:
            raise NotImplementedError(
                f"{e}\nfluxes does not exist\nCreate them by calling `compute_model_fluxes()`"
            )
