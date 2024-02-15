
"""
Deal with the simulations, prepare samplings, and generate instructions
for both fuse (microphysics, detectorphysisc) and WFSim
The core for instructions is taken from Andrii Terliuk's script
"""
import sys
import numpy as np
import pandas as pd
import nestpy, os, click
import warnings
from astropy import units as u
import snewpy
from snewpy.neutrino import Flavor
from scipy.interpolate import interp1d
from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

modules = ['wfsim', 'strax', 'straxen', 'cutax', 'fuse']

module_exists = {}
for module in modules:
    try:
        __import__(module)
        module_exists[module.upper() + 'EXIST'] = True
    except ImportError:
        module_exists[module.upper() + 'EXIST'] = False

WFSIMEXIST = module_exists.get('WFSIMEXIST', False)
STRAXEXIST = module_exists.get('STRAXEXIST', False)
STRAXENEXIST = module_exists.get('STRAXENEXIST', False)
CUTAXEXIST = module_exists.get('CUTAXEXIST', False)
FUSEEXIST = module_exists.get('FUSEEXIST', False)
# default parameters
NEUTRINO_ENERGIES = np.linspace(0,250,500)
RECOIL_ENERGIES = np.linspace(0,30,100)


class SimulationInstructions:
    """ Deal with the FUSE, and WFSim type instructions at different levels.
        Generate and sample features.
        Deal with the context and metadata of the simulation.
    """
    def __init__(self, snax_model=None, snax_interactions=None, simulator='fuse'):
        """ If a snax model is given, obtain the energies and times from that
            else, request energy and time arrays from the user
        """
        self.simulator = simulator
        self.quanta_from_NEST = np.vectorize(self._quanta_from_NEST)
        self.snax_model = snax_model
        self.snax_interactions = snax_interactions
        self.EnergyTimeSampler = EnergyTimeSampling(snax_interactions)

    def generate_vertex(self,
                        r_range=(0, 66.4),
                        z_range=(-148.15, 0), size=1):
        phi = np.random.uniform(size=size) * 2 * np.pi
        r = r_range[1] * np.sqrt(np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size))
        z = np.random.uniform(z_range[0], z_range[1], size=size)
        x = (r * np.cos(phi))
        y = (r * np.sin(phi))
        return x, y, z

    def generate_local_fields(self, pos, fmap=None):
        """ Generate local field values """
        # the interaction sites
        x, y, z = pos
        # getting local field from field map
        if fmap is None:
            fmap = "fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
            print(f"> Using default local field map.: \n> {fmap}")

        if isinstance(fmap, str) and STRAXENEXIST:
            import straxen
            downloader = straxen.MongoDownloader()
            fmap = straxen.InterpolatingMap(
                straxen.get_resource(downloader.download_single(fmap), fmt="json.gz"),
                method="RegularGridInterpolator")
            local_field = fmap(np.array([np.sqrt(x ** 2 + y ** 2), z]).T).repeat(2)
        elif not STRAXENEXIST:
            raise ImportError(f"Straxen is not installed, cannot load the field map: {fmap}")
        elif isinstance(fmap, (float, int)):
            # x has s1 and s2 signals for wfsim, and only 1 entry for fuse
            repeat = len(x) * 2 if self.simulator=='wfsim' else len(x)
            local_field = np.repeat(fmap, repeat)
        else:
            raise TypeError(f"Expected electric field to be either string or float got {type(fmap)}")
        return local_field

    def _get_samples(self, energies, times, pos):
        # get the energies and times of interactions
        if self.snax_interactions is None:
            if energies is None:
                raise ValueError("Please provide energies for the FUSE simulator")
            if times is None:
                raise ValueError("Please provide times for the FUSE simulator")
        else:
            times, neutrino_energy_samples, energies = self.EnergyTimeSampler.sample_times_energies()

        if not isinstance(energies, (list, np.ndarray)):
            raise TypeError(f"Expected energies to be either list or np.ndarray got {type(energies)}")
        if not isinstance(times, (list, np.ndarray)):
            raise TypeError(f"Expected times to be either list or np.ndarray got {type(times)}")
        # get positions
        if pos is None:
            pos = self.generate_vertex(size=len(energies))

        assert len(energies) == len(times) == len(pos[0]), "The number of energies, times and positions should be the same"
        return energies, times, pos

    def generate_fuse_microphysics_instructions(self, times=None, energies=None, pos=None,
                                                interaction_type="neutron", **kwargs):
        """ Generate the microphysics instructions for the FUSE simulator
            If nothing is passed but a snax model is given, it will use the model to generate the instructions
            by default the positions will be uniformly distributed in the TPC
            ["trackid", "parentid", "creaproc", "parenttype", "edproc"] can be passed as kwargs
        """
        # get the energies, times and positions
        energies, times, pos = self._get_samples(energies, times, pos)
        # get the microphysics instructions, quanta is taken care within the fuse
        number_of_events = len(energies)
        instructions = {}
        instructions['eventid'] = np.arange(number_of_events)
        instructions['xp'] = pos[0]
        instructions['yp'] = pos[1]
        instructions['zp'] = pos[2]
        instructions['xp_pri'] = instructions['xp']
        instructions['yp_pri'] = instructions['yp']
        instructions['zp_pri'] = instructions['zp']
        # energy deposition and sampled times
        instructions["ed"] = energies
        instructions['time'] = np.repeat(1, number_of_events)
        instructions["type"] = np.repeat(interaction_type, number_of_events)  # assuming all is NR

        # geant4 related
        instructions["trackid"] = np.zeros(number_of_events)
        instructions["parentid"] = np.zeros(number_of_events, dtype=np.int32)
        instructions["creaproc"] = np.repeat("None", number_of_events)
        instructions["parenttype"] = np.repeat("None", number_of_events)
        instructions["edproc"] = np.repeat("None", number_of_events)
        # Update DataFrame columns based on kwargs
        for column, value in kwargs.items():
            if column in ["trackid", "parentid", "creaproc", "parenttype", "edproc"]:
                if len(value) == number_of_events:
                    instructions[column] = value
                else:
                    warnings.warn(f"Length of {column} does not match the number of events, ignoring")

        return pd.DataFrame(instructions)

    def generate_fuse_detectorphysics_instructions(self, times=None, energies=None, pos=None, fmap=None):
        energies, times, pos = self._get_samples(energies, times, pos)
        number_of_events = len(energies)
        instructions = pd.DataFrame()
        instructions['x'] = pos[0]
        instructions['y'] = pos[1]
        instructions['z'] = pos[2]
        instructions["ed"] = energies
        instructions["t"] = times

        local_fields = self.generate_local_fields(pos, fmap)
        instructions['e_field'] = local_fields

        photons, electrons, excitons = self._quanta_from_NEST(energies, local_fields)
        instructions['photons'] = photons
        instructions['electrons'] = electrons
        instructions["excitons"] = excitons

        instructions["nestid"] = np.array([0] * number_of_events) # 7 is for ER, and 0 is for NR
        instructions["eventid"] = np.arange(number_of_events)
        return instructions

    def generate_wfsim_instructions_from_fuse(self, run_number, times=None, energies=None, pos=None,
                                                interaction_type="neutron", **kwargs):
        """ Generate the wfsim instructions from the FUSE simulator
            Requires FUSE to be installed and the microphysics instructions to be generated
        """
        instructions = self.generate_fuse_microphysics_instructions(times, energies, pos, interaction_type, **kwargs)
        if not FUSEEXIST:
            raise ImportError("FUSE is not installed")
        st = fuse.context.full_chain_context(output_folder="./fuse_data")
        st.register(fuse.micro_physics.output_plugin)
        # try saving the instructions to a file and simulating the microphysics first
        try:
            instructions.to_csv("./fuse_data/temp_microphysics_instructions.csv")
            st.set_config({"path": ".",
                           "file_name": "./fuse_data/temp_microphysics_instructions.csv",
                           "n_interactions_per_chunk": 250,
                           })
            st.make(run_number, "microphysics_summary")
            wfsim_inst = st.get_df(run_number, "wfsim_instructions")
        except Exception as e:
            raise ValueError(f"Failed to generate wfsim instructions from FUSE: {e}")
        return wfsim_inst

    @staticmethod
    def _quanta_from_NEST(en, e_field, **kwargs):
        """
        Simplified version of the fuse function:
        https://github.com/XENONnT/fuse/blob/e6aea3634d96666b1fa8b6e999e082e339ed4660/fuse/plugins/micro_physics/yields.py#L98
        Function which uses NEST to yield photons and electrons
        for a given set of parameters.
        Note:
            In case the energy deposit is outside of the range of NEST a -1
            is returned.
        Args:
            en (numpy.array): Energy deposit of the interaction [keV]
            e_field (numpy.array): Field value in the interaction site [V/cm]
            kwargs: Additional keyword arguments which can be taken by
                GetYields e.g. density.
        Returns:
            photons (numpy.array): Number of generated photons
            electrons (numpy.array): Number of generated electrons
            excitons (numpy.array): Number of generated excitons
        """
        nc = nestpy.NESTcalc(nestpy.VDetector())
        # Some addition taken from
        # https://github.com/NESTCollaboration/nestpy/blob/e82c71f864d7362fee87989ed642cd875845ae3e/src/nestpy/helpers.py#L94-L100
        if en > 2e2:
            print(f"Energy deposition of {en} keV beyond NEST validity for NR model of 200 keV - Remove Interaction")
            return -1, -1, -1

        # should the A, and Z be 131, 54 or 0, 0?
        A, Z = 0, 0 # apparently nestpy handles it internally
        y = nc.GetYields(interaction=nestpy.INTERACTION_TYPE(0),
                         energy=en,
                         drift_field=e_field,
                         A=A,
                         Z=Z,
                         **kwargs
                         )
        event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...
        photons = event_quanta.photons
        excitons = event_quanta.excitons
        electrons = event_quanta.electrons

        return photons, electrons, excitons


class EnergyTimeSampling:
    """ Class to properly sample times and energies from a supernova model
        It also takes care of many-realization sampling.
    """
    def __init__(self, snax_interactions=None):
        self.snax_interactions = snax_interactions

    def generate_times(self, rate, size, timemode='realistic'):
        """ Generate times for the wfsim instructions """
        if timemode == "realistic":
            dt = np.random.exponential(1 / rate, size=size - 1)
            times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
            times = times.round().astype(np.int64)
            return times
        elif timemode == "uniform":
            dt = (1 / rate) * np.ones(size - 1)
            times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
            times = times.round().astype(np.int64)
            return times
        else:
            raise ValueError(f"Time mode {timemode} not supported")

    @staticmethod
    def _inverse_transform_sampling(x_vals, y_vals, n_samples):
        cum_values = np.zeros(x_vals.shape)
        y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
        cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
        inv_cdf = interp1d(cum_values / np.max(cum_values), x_vals)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    def sample_times_energies(self, size='infer', **kw):
        """ Sample interaction times and neutrino energies at those times
            also sample recoil energies based on those neutrino energies and
            the atomic cross-section at that energies.
            :param size: if 'infer' uses the expected number of total counts from the interaction
                        if a single integer, uses the same for each flavor
                        can also be a list of expected counts for each flavor
                        [nue, nue_bar, nux, nux_bar]
            neutrino_energies & recoil_energies can be passed as kwargs
            :returns: sampled_times, sampled_neutrino_energies, sampled_recoils
        """
        time_samples = dict()
        neutrino_energy_samples = dict()
        recoil_energy_samples = dict()
        # check how many interactions to sample from each flavor
        if type(size) == str:
            size = []
            for f in Flavor:
                tot_count = np.trapz(self.snax_interactions.rates_per_recoil_scaled[f],
                                     self.snax_interactions.recoil_energies)
                size.append(int(tot_count.value))
        else:
            if np.ndim(size) == 0:
                size = np.repeat(size, 4)

        # for each flavor sample the times and energies
        for f, s in zip(Flavor, size):
            _time, _nu_energy, _recoil_energy = self._sample_times_energy(self.snax_interactions, s, flavor=f, **kw)
            time_samples[f] = _time
            neutrino_energy_samples[f] = _nu_energy
            recoil_energy_samples[f] = _recoil_energy

        time_samples['Total'] = np.concatenate([time_samples[f] for f in Flavor])
        neutrino_energy_samples['Total'] = np.concatenate([neutrino_energy_samples[f] for f in Flavor])
        recoil_energy_samples['Total'] = np.concatenate([recoil_energy_samples[f] for f in Flavor])

        return time_samples, neutrino_energy_samples, recoil_energy_samples

    def _sample_times_energy(self, interaction, size, flavor=Flavor.NU_E, **kw):
        """
        For the sampling, I could in principle, sample an isotope for each interaction
        based on their abundance. However, this would slow down the sampling heavily,
        therefore, I select always the isotope that has the highest abundance
        """
        # fetch the attributes
        Model = interaction.Model
        times = Model.times.value
        neutrino_energies = kw.get("neutrino_energies", None) or self.snax_interactions.Model.neutrino_energies.value
        recoil_energies = kw.get("recoil_energies", None) or self.snax_interactions.recoil_energies.value
        leave = kw.get("leave", True)
        totrates = self.snax_interactions.rates_per_time_scaled[flavor]

        # sample times
        sampled_times = self._inverse_transform_sampling(times, totrates, size)
        sampled_times = np.sort(sampled_times)

        # fluxes at those times
        # fluxes_at_times = Model.model.get_initial_spectra(t=sampled_times * u.s,
        #                                                   E=neutrino_energies * u.MeV,
        #                                                   flavors=[flavor])[flavor]
        # internal snewpy vectorization error, do it manually
        fluxes_at_times = np.zeros(shape=(len(sampled_times), len(neutrino_energies)))
        # given neutrino flavor, get the fluxes at the sampled times
        for i, j in enumerate(sampled_times):
            fluxes_at_times[i, :] = Model.model.get_initial_spectra(t=j * u.s,
                                                                    E=neutrino_energies * u.MeV,
                                                                    flavors=[flavor])[flavor]
        # get all the cross-sections for a range of neutrino energies
        crosssec = self.snax_interactions.Nucleus[0].nN_cross_section(neutrino_energies * u.MeV,
                                                                      recoil_energies * u.keV)

        # calculate fluxes convolved with this cross-section
        flux_xsec = np.zeros((len(sampled_times), len(recoil_energies), len(neutrino_energies)))
        for i, t in enumerate(sampled_times):
            flux_xsec[i] = (fluxes_at_times[i, :] * crosssec) / np.sum(fluxes_at_times[i, :] * crosssec)

        # select the most abundant atom
        maxabund = np.argmax([nuc.abund for _, nuc in enumerate(self.snax_interactions.Nucleus)])
        atom = self.snax_interactions.Nucleus[maxabund]

        sampled_nues, sampled_recoils = np.zeros(len(sampled_times)), np.zeros(len(sampled_times))
        if not leave:
            pbar, length = enumerate(sampled_times), len(sampled_times)
        else:
            pbar, length = tqdm(enumerate(sampled_times), total=len(sampled_times), desc=flavor.name), None

        for i, t in pbar:
            if length is not None:
                if i / length in [length * 0.25, length * 0.5, length * 0.75]:
                    print(f"\t{i / length:.2%} of {flavor.name} done")

            bb = np.trapz(flux_xsec[i], axis=0)
            sampled_nues[i] = self._inverse_transform_sampling(neutrino_energies, bb, 1)
            recspec = atom.nN_cross_section(sampled_nues[i] * u.MeV, recoil_energies * u.keV).value.flatten()
            sampled_recoils[i] = self._inverse_transform_sampling(recoil_energies, recspec / np.sum(recspec), 1)[0]
        return sampled_times, sampled_nues, sampled_recoils

    def _get_samples_from_model(self):
        """ Get the energies and times from the SNAX model """
        if self.snax_interactions is None:
            raise ValueError("No SNAX model is given")

        # get the energies and times from the model
        # sample times and recoil energies
        time_samples, neutrino_energy_samples, recoil_energy_samples = self.sample_times_energies()
        return time_samples, neutrino_energy_samples, recoil_energy_samples


class MultiSupernovaSimulations(SimulationInstructions):
    """ Class to handle multiple supernova simulations
        Creates individual supernova simulation instructions for each supernova
        Then distributes the times with specified spacings
        Also allows for time-independent instructions to study the signal shape
    """
    def __init__(self, snax_interactions):
        super().__init__(snax_interactions)


    def uniform_time_instructions(self, rate, size, instruction_type='fuse_microphysics', **kw):
        """ Get the instructions for the shape of the signal
            This is useful for the time-independent signal shape
            i.e. the times of the models are ignored
            :rate: the rate of the signal
            :size: the size of the signal
        """
        time_samples, neutrino_energy_samples, recoil_energy_samples= self.EnergyTimeSampler.sample_times_energies(size=size, **kw)
        # overwrite the times with a uniform distribution
        time_samples = self.EnergyTimeSampler.generate_times(rate, size)
        if instruction_type == 'fuse_microphysics':
            instructions = self.generate_fuse_microphysics_instructions(time_samples, recoil_energy_samples, **kw)
        elif instruction_type == 'fuse_detectorphysics':
            instructions = self.generate_fuse_detectorphysics_instructions(time_samples, recoil_energy_samples, **kw)
        elif instruction_type == 'wfsim':
            instructions = self.generate_wfsim_instructions_from_fuse(time_samples, recoil_energy_samples, **kw)
        else:
            raise ValueError(f"Instruction type {instruction_type} not recognized")
        return instructions

    def spaced_time_instructions(self, number_of_supernova,
                                 time_spacing_in_minutes=2,
                                 instruction_type='fuse_microphysics', **kw):
        """ Get the instructions for N supernova signal, spaced in time
            :rate: the rate of the signal
            :size: the size of the signal
        """
        energy_list, time_list = [], []
        for i in range(number_of_supernova):
            # get the energy and time samples from the model
            energies, times, _ = self._get_samples(None, None)
            energy_list.append(energies)
            # add the times with the specified spacing, convert the added times into nanoseconds
            time_list.append(times + i * time_spacing_in_minutes * 60 * 1e9)

        recoil_energy_samples = np.concatenate(energy_list)
        time_samples = np.concatenate(time_list)

        if instruction_type == 'fuse_microphysics':
            instructions = self.generate_fuse_microphysics_instructions(time_samples, recoil_energy_samples, **kw)
        elif instruction_type == 'fuse_detectorphysics':
            instructions = self.generate_fuse_detectorphysics_instructions(time_samples, recoil_energy_samples, **kw)
        elif instruction_type == 'wfsim':
            instructions = self.generate_wfsim_instructions_from_fuse(time_samples, recoil_energy_samples, **kw)
        else:
            raise ValueError(f"Instruction type {instruction_type} not recognized")
        return instructions

class SimulateSignal(MultiSupernovaSimulations):
    """ Use either fuse or wfsim to simulate the signal
        It allows for simulating single or multiple supernova signals
    """
    def __init__(self, snax_interactions, instruction_type='fuse_microphysics'):
        super().__init__(snax_interactions)
        # self.Interaction = snax_interactions
        self.Model = snax_interactions.Model
        self.sim_folder = self.Model.config['wfsim']['sim_folder']
        self.csv_folder = self.Model.config['wfsim']['instruction_path']
        # self.instruction_generator = MultiSupernovaSimulations(snax_interactions)
        self.instruction_type = instruction_type
        self.model_hash = self.snax_interactions.Model.model_hash

    def simulate_single(self, run_number, instructions=None, context=None):
        """ Simulate the signal using the microphysics model """
        # generate and save the instructions
        if instructions is None:
            if self.instruction_type == "fuse_microphysics":
                instructions = self.generate_fuse_microphysics_instructions()
            elif self.instruction_type == "fuse_detectorphysics":
                instructions = self.generate_fuse_detectorphysics_instructions()
            elif self.instruction_type == "wfsim":
                instructions = self.generate_wfsim_instructions_from_fuse(run_number)
            else:
                raise ValueError(f"Instruction type {self.instruction_type} not recognized")

        csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
        instructions.to_csv(f"{self.csv_folder}/{csv_name}", index=False)
        # get the context
        simulator = "fuse" if "fuse" in self.instruction_type else "wfsim"
        st = self.fetch_context(context, simulator)
        # make the simulation
        if self.instruction_type == "fuse_microphysics":
            st.set_config({"path": self.csv_folder,
                           "file_name": csv_name,
                           "n_interactions_per_chunk": 250,
                          })
            st.make(run_number, "microphysics_summary")
        elif self.instruction_type == "fuse_detectorphysics":
            st.register(fuse.detector_physics.ChunkCsvInput)

            st.set_config({"input_file": "./random_detectorphysics_instructions.csv",
                           "n_interactions_per_chunk": 50,
                           })
            st.make(run_number,"raw_records" , progress_bar = True)
        elif self.instruction_type == "wfsim":
            st.set_config(dict(fax_file=f"{self.csv_folder}/{csv_name}"))
            st.make(run_number, "truth")
            st.make(run_number, "raw_records")
        else:
            raise ValueError(f"Instruction type {self.instruction_type} not recognized")

    def simulate_multiple(self, run_number,
                          number_of_supernova=None,
                          rate=None, size=None,
                          time_spacing_in_minutes=None,
                          context=None):
        """ Simulate multiple supernova signals
            If (number of supernova, time spacing) is provided, simulate that many signals,
            properly spaced in time and using actual model times
            else if (rate, size) is provided, simulate the signals with that rate and size
            ignore the model times and simulate a uniform distribution of signals in time
        """
        is_time_independent = (rate is not None) and (size is not None)
        is_time_dependent = (number_of_supernova is not None) and (time_spacing_in_minutes is not None)

        if not is_time_dependent and not is_time_independent:
            raise ValueError("Please provide either number_of_supernova and time_spacing_in_minutes or rate and size")

        if is_time_independent:
            instructions = self.uniform_time_instructions(rate, size, self.instruction_type)
        else:
            instructions = self.spaced_time_instructions(run_number, time_spacing_in_minutes)

        self.simulate_single(run_number, instructions=instructions, context=context)


    def fetch_context(self, context, simulator):
        """ Fetch the context for the simulation
            If context is updated, change it in here
            Requires config to be a configparser object with ['wfsim']['sim_folder'] field
            So that the strax data folder can be found
        """
        data_folder = "fuse_data" if simulator=="fuse" else "strax_data"
        mc_data_folder = os.path.join(self.sim_folder, data_folder)

        def _add_strax_directory(context):
            # add the mc folder and return it
            output_folder_exists = False
            # check if the mc folder is already in the context
            for i, stores in enumerate(context.proc_loc):
                if mc_data_folder in stores.path:
                    output_folder_exists = True
            # if it is not yet in the context, add a DataDirectory
            if not output_folder_exists:
                if not STRAXEXIST:
                    raise ImportError("strax not installed")
                import strax
                context.proc_loc += [strax.DataDirectory(mc_data_folder, readonly=False)]

        # if a context is given, check if the storage is correct
        if context is not None:
            _add_strax_directory(context)
            return context
        # if no context is given, create a new one
        if simulator == 'wfsim':
            if not WFSIMEXIST or not CUTAXEXIST:
                raise ImportError("wfsim or cutax not installed")
            import cutax
            context = cutax.contexts.xenonnt_sim_SR0v4_cmt_v9(output_folder=mc_data_folder)
        elif simulator == 'fuse':
            if not FUSEEXIST:
                raise ImportError("fuse not installed")
            import fuse
            context = fuse.context.full_chain_context(output_folder=mc_data_folder)
        else:
            raise ValueError(f"Simulator {simulator} not recognized")
        # add the strax folder to the context
        _add_strax_directory(context)
        return context


