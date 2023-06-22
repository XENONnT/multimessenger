#!/usr/bin/python
import json
import warnings
import numpy as np
import pandas as pd
import astropy
from astropy import units as u
from snewpy.neutrino import Flavor
import copy
from .sn_utils import isnotebook, see_simulated_contexts
from .Nucleus import Target
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import click, os, json
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
customstyle = os.path.join(os.path.dirname(os.path.realpath(__file__)), "customstyle.mplstyle")
plt.style.use(customstyle)
# plt.style.use('customstyle.mplstyle')

class InteractionSingle:
    def __init__(self, Model, Target, recoil_energies):
        # read the SN model and the time and neutrino energies
        self.Model = Model
        self.name = Target.name
        self.times = self.Model.times
        self.neutrino_energies = self.Model.neutrino_energies
        # read the single isotope target and requested recoil energies
        self.Target = Target
        self.recoil_energies = recoil_energies
        # get cross-section for a given recoil and neutrino energy (len(E_nu), len(E_R))
        self.xsecs = self.Target.nN_cross_section(self.neutrino_energies, self.recoil_energies)

    def dRdEr(self):
        """ Return rates per given recoil energies
            time integrated
            Returns
              $\frac{dR}{dE_R} = \sum_{\nu_\beta} N_{Xe}
               \int_{E_{min}^{\nu}} dE_\nu f_\nu(E_\nu)\frac{d\sigma}{dE_R}(E_\nu, E_R)$
            So the result is a dictionary with matrices
        """
        # get time integrated fluxes (len(E_nu))
        time_integrated_fluxes = {f: np.trapz(self.Model.fluxes[f], self.times, axis=0).to(1 / u.keV) for f in Flavor}
        # calculate the discrete integrand f(Enu)*cross_sec
        integrand = {f: (self.xsecs * time_integrated_fluxes[f]) for f in Flavor}
        # integrate over neutrino energies and scale by the isotope abundance
        rates_per_recoil_single = {f: self.Target.abund * np.trapz(integrand[f], self.neutrino_energies, axis=1) for f in
                            Flavor}
        # change the units and return
        rates_per_recoil_single = {k: v.to(u.m ** 2 / u.keV) for k, v in rates_per_recoil_single.items()}
        return rates_per_recoil_single

    def dRdt(self):
        """ Return rates per time
            Integrated over neutrino energies, notice this wouldn't affect the
            neutrino flux. So we deal with the cross-section part
              $\frac{dR}{dt} = \sum N_{Xe}\int_{E_{min}^\nu} = f_\nu(E_\nu,t)
               \int_{E_{R}^{min}}^{E_{R}^{max}} dE_R \frac{d\sigma}{dE_R}(E_\nu, E_R)$
        """
        # integrate the cross-section matrix over recoil energies
        xsecs_integrated = np.trapz(self.xsecs, self.recoil_energies, axis=0)
        # multiply the 1D fluxes with this now-1D cross-section
        integrand = {f: self.Model.fluxes[f] * xsecs_integrated for f in Flavor}
        # integrate over neutrino energies
        rates_per_time_single = {f: self.Target.abund * np.trapz(integrand[f], self.neutrino_energies, axis=1) for f in
                            Flavor}
        # change the units and return
        rates_per_time_single = {k:v.to(u.m ** 2 / u.s) for k,v in rates_per_time_single.items()}
        return rates_per_time_single


class Interactions:
    def __init__(self, Model, Nuclei="Xenon", isotope="mix"):
        """ Set a Supernova Model and a Nuclei to interact
            Model is a Supernova_Models.Models object
            If isotope is mix, then a combination of isotopes is used with
            abundances used as weights.
            Possible isotopes; [Xe124, Xe126, Xe128, Xe129, Xe130, Xe131,
                                Xe132, Xe134, Xe136]
        """
        self.Model = Model
        self.interaction_file = ".".join(self.Model.object_name.split('.')[:-1])+"_interaction.pickle"
        if Nuclei=="Xenon":
            from .Xenon_Atom import ATOM_TABLE
        else:
            raise NotImplementedError(f"Requested {Nuclei} but only have Xenon for now")
        if isotope=='mix':
            self.Nucleus = [Target(ATOM_TABLE["Xe124"], pure=False),
                            Target(ATOM_TABLE["Xe126"], pure=False),
                            Target(ATOM_TABLE["Xe128"], pure=False),
                            Target(ATOM_TABLE["Xe129"], pure=False),
                            Target(ATOM_TABLE["Xe130"], pure=False),
                            Target(ATOM_TABLE["Xe131"], pure=False),
                            Target(ATOM_TABLE["Xe132"], pure=False),
                            Target(ATOM_TABLE["Xe134"], pure=False),
                            Target(ATOM_TABLE["Xe136"], pure=False)]
        else:
            self.Nucleus = [Target(ATOM_TABLE[isotope], pure=True)]

        self.Nuclei_name = Nuclei
        self.isotope_name = isotope
        self.recoil_energies = np.linspace(0,20,100) * u.keV
        self.all_targets = [InteractionSingle(self.Model, t, self.recoil_energies) for t in self.Nucleus]
        self.rates_per_recoil_iso = None
        self.rates_per_time_iso = None
        self.rates_per_recoil = None
        self.rates_per_time = None
        # after scale
        self.rates_per_recoil_scaled = None
        self.rates_per_time_scaled = None
        self.volume = None
        self.distance = None
        self.expected_total = dict()

        if os.path.isfile(self.interaction_file):
            # try to retrieve
            self.retrieve_object(self.interaction_file)
        else:
            self.save_object(update=True)

    def save_object(self, update=False):
        """ Save the object for later calls
        """
        if update:
            full_file_path = os.path.join(self.Model.storage, self.interaction_file)
            with open(full_file_path, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(self, output, -1)  # pickle.HIGHEST_PROTOCOL
                click.secho(f'> Saved at <self.storage>/{self.interaction_file}!\n', fg='green')

    def retrieve_object(self, name=None):
        file = name or self.interaction_file
        full_file_path = os.path.join(self.Model.storage, file)
        with open(full_file_path, 'rb') as handle:
            click.secho(f'> Retrieving object self.storage/{file}', fg='blue')
            tmp_dict = pickle.load(handle)
        self.__dict__.update(tmp_dict.__dict__)
        return None

    def delete_object(self):
        full_file_path = os.path.join(self.Model.storage, self.interaction_file)
        if input(f"> Are you sure you want to delete\n"
                 f"{full_file_path}?\n") == 'y':
            os.remove(full_file_path)

    def __repr__(self):
        """Default representation of the model.
        """
        _repr = self.Model.__repr__()
        return _repr

    def _repr_markdown_(self):
        """Markdown representation of the model, for Jupyter notebooks.
        """
        try:
            _repr = self.Model._repr_markdown_()
        except AttributeError:
            _repr = f"**{self.Model.object_name}**"

        s = [_repr]
        s += [f"|Interaction file| {self.interaction_file}"]
        s += [f"|Target | {self.isotope_name} {self.Nuclei_name}"]
        is_computed = type(self.rates_per_time) != type(None)
        is_scaled = type(self.rates_per_recoil_scaled) != type(None)
        s += [f"|Computed, scaled | {is_computed}, {is_scaled}"]
        if is_scaled:
            s += [f"|distance | {self.distance}"]
            s += [f"|volume | {self.volume}"]
            s += [f"|Expected Total | {self.expected_total['Total']:.0f}"]
        return '\n'.join(s)

    def compute_interaction_rates(self, recoil_energies=None, force=False, **kw_model):
        """
        :param recoil_energies: `array` default 0-20 keV 100 samples
            the recoil energies at which the interactions are calculated
            if no unit is used, assumes keV
        **kw_model keyword arguments for the model in order to compute the fluxes
        see Supernova_Models.Models.compute_model_fluxes()
        Only used if the fluxes are not computed for the model that is passed
        """
        recoil_energies = recoil_energies or self.recoil_energies
        if not type(recoil_energies) == u.quantity.Quantity:
            # assume keV
            recoil_energies *= u.keV

        # check if the model fluxes are computed
        if self.Model.fluxes is None:
            click.secho(f"\t> Fluxes is not computed for {self.Model.object_name}, \n\tcomputing now...", fg='blue')
            self.Model.compute_model_fluxes(**kw_model)

        if self.rates_per_recoil is not None and not force:
            click.secho(f"\t> Recoil rates have been found! Stored in self.rates_per_recoil & self.rates_per_time")
            # rates have been computed
            return None

        # interaction rates for each isotope
        self.rates_per_recoil_iso = {isotope.name: isotope.dRdEr() for isotope in tqdm(self.all_targets)}
        self.rates_per_time_iso = {isotope.name: isotope.dRdt() for isotope in tqdm(self.all_targets)}

        # sum over the isotopes and get the rates for the nuclei, creates self.rates_per_time(recoil)
        self._compute_total_rates()


    def _compute_total_rates(self):
        """ Sum over all isotopes the get total rates per flavor
            Then also sum over all flavor to get the actual total.
        """
        # get the total fluxes and rates
        dEr_example = self.rates_per_recoil_iso[self.all_targets[0].name][Flavor.NU_E]
        dt_example = self.rates_per_time_iso[self.all_targets[0].name][Flavor.NU_E]

        self.rates_per_recoil = {f: np.zeros_like(dEr_example) for f in Flavor}
        self.rates_per_time = {f: np.zeros_like(dt_example) for f in Flavor}
        for f in Flavor:
            for xe in self.all_targets:
                self.rates_per_recoil[f] += self.rates_per_recoil_iso[xe.name][f]
                self.rates_per_time[f] += self.rates_per_time_iso[xe.name][f]

        # add the total
        self.rates_per_recoil['Total'] = np.zeros_like(self.rates_per_recoil[Flavor.NU_E])
        self.rates_per_time['Total']  = np.zeros_like(self.rates_per_time[Flavor.NU_E])
        for f in Flavor:
            _rate1 = self.rates_per_recoil[f]
            _rate2 = self.rates_per_time[f]
            if not f.is_electron:
                _rate1 *=  2  # x-neutrinos are for muon and tau
                _rate2 *=  2 # x-neutrinos are for muon and tau
            self.rates_per_recoil['Total'] += _rate1
            self.rates_per_time['Total'] += _rate2

        click.secho(f"> Computed the total rates at the source for 1 atom (not scaled)", fg='blue')


    # def truncate_rates(self):
    #     """ Truncate rates and recoil energies and times
    #     Returns: rates_Er_tr, rates_t_tr, recen_tr, times_tr
    #     """
    #     recoil_energies = self.recoil_energies
    #     rates_Er = self.rateper_Er
    #     times = self.model.time
    #     rates_t = self.rateper_t
    #
    #     # truncate all at position where total rate becomes 0
    #     # rateper_Er
    #     trunc_here = np.searchsorted(rates_Er['Total'][::-1], 0, side='right')
    #     rates_Er_tr = {nu: rates_Er[nu][:-trunc_here] for nu in rates_Er.keys()}
    #     recen_tr = recoil_energies[:-trunc_here]
    #     # rateper_t
    #     trunc_here = np.searchsorted(rates_t['Total'][::-1], 0, side='right')
    #     rates_t_tr = {nu: rates_t[nu][:-trunc_here] for nu in rates_t.keys()}
    #     times_tr = times[:-trunc_here]
    #     return rates_Er_tr, rates_t_tr, recen_tr, times_tr

    def _compute_expected_total(self):
        """ for given scaled rates compute the total expected counts
        """
        for k, v in self.rates_per_recoil_scaled.items():
            self.expected_total[k] = np.trapz(v, self.recoil_energies)



    def scale_rates(self, distance, volume, N_Xe=4.6e27*u.count/u.tonne):
        """ Scale the rates  based on distance and number of atoms
            distance is assumed to be given in kpc or with units

            Returns: `tuple` (scaled rates per recoil, scaled rates per time)
        """
        # each time copy from the rates at th source
        self.rates_per_recoil_scaled = copy.deepcopy(self.rates_per_recoil)
        self.rates_per_time_scaled = copy.deepcopy(self.rates_per_time)
        if not type(distance) == u.quantity.Quantity:
            # assume kpc
            distance *= u.kpc
        if not type(volume) == u.quantity.Quantity:
            # assume ton
            volume *= u.tonne
        # track the parameters
        self.volume = volume
        self.distance = distance

        scale = N_Xe / (4 * np.pi * distance ** 2).to(u.m ** 2)
        scale *= volume
        try:
            for f in self.rates_per_time_scaled.keys():
                self.rates_per_recoil_scaled[f] *= scale
                self.rates_per_time_scaled[f] *= scale
                # self.expected_total
            self._compute_expected_total()
            return self.rates_per_recoil_scaled, self.rates_per_time_scaled
        except:
            raise NotImplementedError("\t> Rates haven't been computed yet!\n"
                                      "\t>Compute them by calling `compute_interaction_rates()`")

    def simulate_automatically(self, runid, context=None, config=None, return_instructions=False, force=False, **kw):
        """ Simulate using WFSim automatically from your interactions
            First sample times and energies
            Next, generate sn instructions
            Finally, simulate a single WFSim simulation from the instructions
            kwargs can include:
            :param nc: `nestpy.NESTcalc()` instance
            :param fmap: `str` name of the electric field file, default
                "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
                or `straxen.InterpolatingMap`
            :param field: `float` if no map is passed, fixed field
            :param force: `bool` even if data exists, recreate from scratch

            Returns:
            context, (instructions)
        """
        from .Simulate import  _simulate_one, sample_times_energies, generate_sn_instructions
        from .sn_utils import add_strax_folder, fetch_context
        config = config or self.Model.config
        if context is None:
            _context = fetch_context(config) # returns default context with output folder pointing to common strax data folder
        else:
            _context = add_strax_folder(config, context) # to have access to common strax data folder

        # sample times and recoil energies
        time_samples, _, recoil_energy_samples = sample_times_energies(self, size='infer')

        # default field file
        field_file = "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"

        instructions = generate_sn_instructions(energy_deposition=recoil_energy_samples['Total'],
                                                n_tot=len(time_samples['Total']),  # times in ns
                                                times=time_samples['Total'] * 1e9,
                                                fmap=field_file,
                                                **kw)
        instructions = pd.DataFrame(instructions)
        st = _simulate_one(instructions, runid, config=config, context=_context, force=force)
        to_return = st

        try:
            self._make_json(runid, config)
        except Exception as e:
            print(f">>> Problem making an entry to the JSON {runid}\n{e}\n")

        # Maybe no need to return the context again?
        if return_instructions:
            to_return = [st, instructions]
        return to_return

    def simulate_many(self, context, runid, config=None,
                      return_instructions=False,
                      force=False,
                      N_supernova=200,
                      shift_method='random',
                      **kw):
        """ Simulate many SN using WFSim
            see also simulate_automatically()
            :param N_supernova: `int` number of supernova to simulate
            :param shift_method: `str` random, oneafterother

            Returns:
            context, (instructions)
        """
        from .Simulate import  _simulate_one, sample_times_energies, generate_sn_instructions
        from .sn_utils import add_strax_folder
        config = config or self.Model.config
        _context = add_strax_folder(config, context) # to have access to common strax data folder

        # sample times and recoil energies
        duration = np.ptp(self.Model.model.time).value
        if shift_method=='random':
            shifts = np.random.uniform(0, 10, N_supernova)
        elif shift_method=='oneafterother':
            shifts = np.repeat(duration, N_supernova)
        else:
            warnings.warn(f"The shift_method={shift_method} is not recognized, shifting randomly")
            shifts = np.random.uniform(0, 10, N_supernova)

        max_time = 0
        _, _, foo = sample_times_energies(self, size='infer', leave=False)
        single_sample_size = len(foo['Total'])
        time_samples = np.zeros(single_sample_size*N_supernova, dtype=np.float32)
        recoil_energy_samples = np.zeros(single_sample_size*N_supernova, dtype=np.float32)
        identifier = np.zeros(single_sample_size*N_supernova, dtype=int)
        for i in range(N_supernova):
            time_sample, _, recoil_energy_sample = sample_times_energies(self, size='infer', leave=False)
            time_sample, recoil_energy_sample = time_sample['Total'], recoil_energy_sample['Total']
            # adjust arrays
            _from = int(i * single_sample_size)
            _to = int((i + 1) * single_sample_size)
            recoil_energy_samples[_from:_to] = recoil_energy_sample
            time_samples[_from:_to] = time_sample
            time_samples[_from:_to] += max_time   # shift by the max time so that each SN starts at a later time. (ensure no overlap)
            time_samples[_from:_to] += shifts[i]  # add the requested shift.
            max_time = np.max(time_samples)      # max time of *all* registered times
            identifier[_from:_to] = i

        # default field file
        field_file = "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
        instructions, nonzeromask = generate_sn_instructions(energy_deposition=recoil_energy_samples,
                                                             n_tot=len(time_samples),  # times in ns
                                                             times=time_samples * 1e9,
                                                             fmap=field_file,
                                                             return_nonzero_mask=True,
                                                             **kw)
        instructions = pd.DataFrame(instructions)
        st = _simulate_one(instructions, runid, config=config, context=_context, force=force)
        to_return = st
        if return_instructions:
            # also add the identifier
            instructions['identifier'] = identifier.repeat(2)[nonzeromask]
            to_return = [st, instructions]
        return to_return

    def _make_json(self, sim_id, config, jsonfilename="simulation_metadata.json"):
        """ Make a json file that contains the metadata of the simulation
        """
        model = self.Model
        snewpymodel = model.model
        # where to save the json file
        try:
            store_at = model.config['wfsim']['sim_folder']
        except Exception as e:
            print(f"WFSim / sim_folder could not be found, storing the metadata in cwd,\n{e}")
            store_at = "./"
        # Check if json exists, create if not
        output_json = os.path.join(store_at, jsonfilename)
        # os.makedirs(output_json, exist_ok=True)

        # create some metadata
        meta = {'User': model.user, 'Storage': model.storage, 'Model Name': model.model_name,
                'Sim File': model.object_name,
                'Time Range': f"{model.time_range[0]}, {model.time_range[1]}"}
        # metadata from the snewpy model
        for k, v in snewpymodel.metadata.items():
            if isinstance(v, astropy.units.quantity.Quantity):
                v = f"{v}"
            meta[k] = v
        meta['Model File'] = getattr(snewpymodel, "filename", "Unknown Snewpy Model Name")
        meta['Duration'] = f"{np.round(np.ptp(snewpymodel.time), 2)}"
        # metadata from the interaction object
        meta['Interaction File'] = self.interaction_file
        meta['Nuclei Name'] = self.Nuclei_name
        meta['Isotope Name'] = self.isotope_name
        # metadata from the wfsim context
        df = see_simulated_contexts(config_file=config, sim_id=sim_id)
        df_dict = df.iloc[0].to_dict()
        df_dict['context_name'] = df_dict['name']
        df_dict['date_added'] = f"{df_dict['date_added']}"
        df_dict.pop('sim_id')
        df_dict.pop('name')
        # make a json entry
        json_entry = {sim_id: {"Model": meta, "Context": df_dict}}
        # Append this simulation
        if os.path.exists(output_json):
            # read existing file and append new data
            with open(output_json, "r") as f:
                dictObj = json.load(f)
            dictObj.update(json_entry)
        else:
            # create new json
            dictObj = json_entry

        # overwrite/create file
        with open(output_json, "w") as f:
            json.dump(dictObj, f, indent=4, sort_keys=True)

    def plot_rates(self, scaled=True):
        """ Plot the rates, scaled or total
        """
        if scaled:
            if type(self.rates_per_time_scaled)==type(None):
                click.secho(f"\t>Scaled rates have not been computed!, Use self.scale_rates()", fg='red')
                return None
            rates_time = self.rates_per_time_scaled
            rates_recoil = self.rates_per_recoil_scaled
            d = f"{self.distance.value} {self.distance.unit}"
            v = f"{self.volume.value} {self.volume.unit}"
        else:
            if type(self.rates_per_recoil)==type(None):
                click.secho(f"\t>Rates have not been computed!, Use self.compute_interaction_rates()", fg='red')
                return None
            rates_time = self.rates_per_time
            rates_recoil = self.rates_per_recoil
            d = "source"
            v = "1 atom"

        times = self.Model.times
        recoils = self.recoil_energies
        # plot the rates
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        for f in Flavor:
            ax1.semilogx(times, rates_time[f])
            ax2.plot(recoils, rates_recoil[f], label=f.to_tex())

        ax1.semilogx(times, rates_time['Total'], color='k')
        ax2.plot(recoils, rates_recoil['Total'], color='k', label='Total')

        ax1.set_ylabel(f"Rates [{rates_time['Total'].unit}]\n (at {d}, for {v})")
        ax1.set_xlabel(f'times [{times.unit}]')
        ax2.set_ylabel(f"Rates [{rates_recoil['Total'].unit}]\n (at {d}, for {v})")
        ax2.set_xlabel(f'Recoil Energies [{recoils.unit}]')
        ax2.legend()