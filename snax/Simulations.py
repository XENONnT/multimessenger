"""
Deal with the simulations, prepare samplings, and generate instructions
for both fuse (microphysics, detectorphysisc) and WFSim
The core for instructions is taken from Andrii Terliuk's script
"""

import numpy as np
import pandas as pd
import nestpy, os
import warnings
from astropy import units as u
from snewpy.neutrino import Flavor
from scipy.interpolate import interp1d
from .sn_utils import isnotebook

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

modules = ["wfsim", "strax", "straxen", "cutax", "fuse"]

module_exists = {}
for module in modules:
    try:
        __import__(module)
        module_exists[module.upper() + "EXIST"] = True
    except ImportError:
        module_exists[module.upper() + "EXIST"] = False

WFSIMEXIST = module_exists.get("WFSIMEXIST", False)
STRAXEXIST = module_exists.get("STRAXEXIST", False)
STRAXENEXIST = module_exists.get("STRAXENEXIST", False)
CUTAXEXIST = module_exists.get("CUTAXEXIST", False)
FUSEEXIST = module_exists.get("FUSEEXIST", False)
# default parameters
NEUTRINO_ENERGIES = np.linspace(0, 250, 500)
RECOIL_ENERGIES = np.linspace(0, 30, 100)

DEFAULT_XEDOCS_VERSION = 'global_v14'
DEFAULT_SIMULATION_VERSION = 'fax_config_nt_sr0_v4.json'

# all params file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, os.pardir)
allparams_csv_path = os.path.join(parent_dir, 'all_parameters.csv')


class SimulationInstructions:
    """Deal with the FUSE, and WFSim type instructions at different levels.
    Generate and sample features.
    Deal with the context and metadata of the simulation.
    """

    def __init__(self, snax_interactions=None, simulator="fuse"):
        """If a snax interactions is given, obtain the energies and times from that
        else, request energy and time arrays from the user
        """
        self.simulator = simulator
        self.quanta_from_NEST = np.vectorize(self._quanta_from_NEST)
        self.snax_interactions = snax_interactions
        self.EnergyTimeSampler = EnergyTimeSampling(snax_interactions)
        is_computed = snax_interactions.rates_per_time is not None
        is_scaled = snax_interactions.rates_per_recoil_scaled is not None
        if not is_computed or not is_scaled:
            warnings.warn(
                f"The interaction rates are {'' if is_computed else 'not'} computed, "
                f"and rates are {'' if is_scaled else 'not'} scaled!\n"
                f"interaction rates need to be computed first!"
            )

    def generate_vertex(self, r_range=(0, 66.4), z_range=(-148.15, 0), size=1):
        phi = np.random.uniform(size=size) * 2 * np.pi
        r = r_range[1] * np.sqrt(
            np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size)
        )
        z = np.random.uniform(z_range[0], z_range[1], size=size)
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y, z

    def generate_local_fields(self, pos, fmap=None):
        """Generate local field values"""
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
                method="RegularGridInterpolator",
            )
            local_field = fmap(np.array([np.sqrt(x**2 + y**2), z]).T)
            # wfsim has 2 entries per interaction while fuse has 1
            local_field = local_field if self.simulator != "wfsim" else local_field.repeat(2)
        elif not STRAXENEXIST:
            raise ImportError(
                f"Straxen is not installed, cannot load the field map: {fmap}"
            )
        elif isinstance(fmap, (float, int)):
            # x has s1 and s2 signals for wfsim, and only 1 entry for fuse
            repeat = len(x) * 2 if self.simulator == "wfsim" else len(x)
            local_field = np.repeat(fmap, repeat)
        else:
            raise TypeError(
                f"Expected electric field to be either string or float got {type(fmap)}"
            )
        return local_field

    def _get_samples(self, times, energies, pos):
        """returns the times in seconds!
        If energies and times are provided, return them back
        If not, then sample by inferring the size
        """
        # get the energies and times of interactions
        if self.snax_interactions is None:
            if energies is None or times is None:
                raise ValueError(
                    "Please provide both energies and times for the FUSE simulator."
                )
        else:
            # if the interactions exists and either of times and energies passed, sample
            if energies is None or times is None:
                # Sample energies and times if not provided, times are sampled in seconds!
                _times, neutrino_energy_samples, _energies = (
                    self.EnergyTimeSampler.sample_times_energies(return_totals=True)
                )
                energies = energies if energies is not None else _energies
                times = times if times is not None else _times

        if not isinstance(energies, (list, np.ndarray)):
            raise TypeError(
                f"Expected energies to be either list or np.ndarray got {type(energies)}"
            )
        if not isinstance(times, (list, np.ndarray)):
            raise TypeError(
                f"Expected times to be either list or np.ndarray got {type(times)}"
            )
        # get positions
        if pos is None:
            pos = self.generate_vertex(size=len(energies))

        assert (
            len(energies) == len(times) == len(pos[0])
        ), "The number of energies, times and positions should be the same"
        return times, energies, pos

    def generate_fuse_microphysics_instructions(
        self, times=None, energies=None, pos=None, interaction_type="neutron", **kwargs
    ):
        """Generate the microphysics instructions for the FUSE simulator
        If nothing is passed but a snax model is given, it will use the model to generate the instructions
        by default the positions will be uniformly distributed in the TPC
        ["trackid", "parentid", "creaproc", "parenttype", "edproc"] can be passed as kwargs
        """
        # get the energies, times (in seconds) and positions
        times, energies, pos = self._get_samples(times, energies, pos)
        # convert the times into nanosec
        times = times * 1e9 # A very weird bug if times *= 1e9
        times = times.round().astype(np.int64)
        # get the microphysics instructions, quanta is taken care within the fuse
        number_of_events = len(energies)
        instructions = {}
        instructions["eventid"] = np.arange(number_of_events)
        instructions["xp"] = pos[0]
        instructions["yp"] = pos[1]
        instructions["zp"] = pos[2]
        instructions["xp_pri"] = instructions["xp"]
        instructions["yp_pri"] = instructions["yp"]
        instructions["zp_pri"] = instructions["zp"]
        # energy deposition and sampled times
        instructions["ed"] = energies
        instructions["time"] = times
        instructions["type"] = np.repeat(
            interaction_type, number_of_events
        )  # assuming all is NR

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
                    warnings.warn(
                        f"Length of {column} does not match the number of events, ignoring"
                    )

        instructions = pd.DataFrame(instructions)
        # microphysics instructions use mm instead of cm
        for col in ['xp', 'yp', 'zp', 'xp_pri', 'yp_pri', 'zp_pri']:
            instructions.loc[:, col] *= 10

        return instructions

    def generate_fuse_detectorphysics_instructions(
        self, times=None, energies=None, pos=None, fmap=None
    ):
        # get the energies, times (in seconds) and positions
        times, energies, pos = self._get_samples(times, energies, pos)
        # convert the times into nanosec
        times = times * 1e9 # A very weird bug if times *= 1e9
        times = times.round().astype(np.int64)
        number_of_events = len(energies)
        instructions = pd.DataFrame()
        instructions["x"] = pos[0]
        instructions["y"] = pos[1]
        instructions["z"] = pos[2]
        instructions["ed"] = energies
        instructions["t"] = times

        local_fields = self.generate_local_fields(pos, fmap)
        instructions["e_field"] = local_fields

        photons, electrons, excitons = self.quanta_from_NEST(energies, local_fields)
        instructions["photons"] = photons
        instructions["electrons"] = electrons
        instructions["excitons"] = excitons

        instructions["nestid"] = np.array(
            [0] * number_of_events
        )  # 7 is for ER, and 0 is for NR
        instructions["eventid"] = np.arange(number_of_events)
        return instructions

    def generate_wfsim_instructions_from_fuse(
        self,
        run_number,
        times=None,
        energies=None,
        pos=None,
        interaction_type="neutron",
        **kwargs,
    ):
        """Generate the wfsim instructions from the FUSE simulator
        Requires FUSE to be installed and the micro-physics instructions to be generated
        """
        instructions = self.generate_fuse_microphysics_instructions(
            times, energies, pos, interaction_type, **kwargs
        )
        if not FUSEEXIST:
            raise ImportError("FUSE is not installed")
        import fuse
        st = fuse.context.full_chain_context(output_folder="./fuse_data")
        st.register(fuse.micro_physics.output_plugin)
        # try saving the instructions to a file and simulating the microphysics first
        try:
            instructions.to_csv("./fuse_data/temp_microphysics_instructions.csv")
            st.set_config(
                {
                    "path": ".",
                    "file_name": "./fuse_data/temp_microphysics_instructions.csv",
                    "n_interactions_per_chunk": 250,
                }
            )
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
            print(
                f"Energy deposition of {en} keV beyond NEST validity for NR model of 200 keV - Remove Interaction"
            )
            return -1, -1, -1

        # should the A, and Z be 131, 54 or 0, 0?
        A, Z = 0, 0  # apparently nestpy handles it internally
        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(0),
            energy=en,
            drift_field=e_field,
            A=A,
            Z=Z,
            **kwargs,
        )
        event_quanta = nc.GetQuanta(y)  # Density argument is not use in function...
        photons = event_quanta.photons
        excitons = event_quanta.excitons
        electrons = event_quanta.electrons

        return photons, electrons, excitons

    def uniform_time_instructions(
        self, rate, size, instruction_type="fuse_microphysics", run_number="00000", **kw
    ):
        """Get the instructions for the shape of the signal
        This is useful for the time-independent signal shape
        i.e. the times of the models are ignored
        :rate: the rate of the signal i.e. entry per sec
        :size: the size of the signal i.e. total entry
        :param run_number: run number, required for wfsim instructions
        Notes:
            here the times are given in seconds, the respective `generate_XXX` function converts them to ns
            example; rate=10, size=50 would return 50 data points spread within ~5 sec
        """
        _, _, recoil_energy_samples = self.EnergyTimeSampler.sample_times_energies(
            size=size, **kw
        )
        # the function samples size*4 for each flavor, since times are uniform and random, sample equal amount of E_r
        recoil_energy_samples = np.random.choice(recoil_energy_samples, size)

        # overwrite the times with a uniform distribution
        time_samples = self.EnergyTimeSampler.generate_times(
            rate, size
        )  # here in seconds, below converted to ns
        if instruction_type == "fuse_microphysics":
            instructions = self.generate_fuse_microphysics_instructions(
                time_samples, recoil_energy_samples, **kw
            )
        elif instruction_type == "fuse_detectorphysics":
            instructions = self.generate_fuse_detectorphysics_instructions(
                time_samples, recoil_energy_samples, **kw
            )
        elif instruction_type == "wfsim":
            instructions = self.generate_wfsim_instructions_from_fuse(
                run_number, time_samples, recoil_energy_samples, **kw
            )
        else:
            raise ValueError(f"Instruction type {instruction_type} not recognized")
        return instructions

    def spaced_time_instructions(
        self,
        number_of_supernova,
        time_spacing_in_minutes=2,
        instruction_type="fuse_microphysics",
        run_number="00000",
        **kw,
    ):
        """Get the instructions for N supernova signal, spaced in time
        :rate: the rate of the signal
        :size: the size of the signal
        """
        energy_list, time_list = [], []
        for i in tqdm(range(number_of_supernova), total=number_of_supernova):
            # get the energy and time samples (in seconds) from the model
            times, _, energies = self.EnergyTimeSampler.sample_times_energies(
                return_totals=True, **kw
            )
            energy_list.append(energies)
            time_list.append(
                times + i * time_spacing_in_minutes * 60
            )  # here shift in seconds! Later converted to ns

        recoil_energy_samples = np.concatenate(energy_list)
        time_samples = np.concatenate(time_list)

        # generate instructions, notice times are converted to ns within the generate function
        if instruction_type == "fuse_microphysics":
            instructions = self.generate_fuse_microphysics_instructions(
                time_samples, recoil_energy_samples, **kw
            )
        elif instruction_type == "fuse_detectorphysics":
            instructions = self.generate_fuse_detectorphysics_instructions(
                time_samples, recoil_energy_samples, **kw
            )
        elif instruction_type == "wfsim":
            instructions = self.generate_wfsim_instructions_from_fuse(
                run_number, time_samples, recoil_energy_samples, **kw
            )
        else:
            raise ValueError(f"Instruction type {instruction_type} not recognized")
        return instructions


class EnergyTimeSampling:
    """Class to properly sample times and energies from a supernova model
    It also takes care of many-realization sampling.
    """

    def __init__(self, snax_interactions=None):
        self.snax_interactions = snax_interactions

    def generate_times(self, rate, size, timemode="realistic"):
        """Generate times for the wfsim instructions
        times returned are in seconds
        """
        if timemode == "realistic":
            dt = np.random.exponential(1 / rate, size=size - 1)
        elif timemode == "uniform":
            dt = (1 / rate) * np.ones(size - 1)
        else:
            raise ValueError(f"Time mode {timemode} not supported")
        times = np.append([1.0], 1.0 + dt.cumsum())
        # times = times.round().astype(np.int64)
        return times

    @staticmethod
    def _inverse_transform_sampling(x_vals, y_vals, n_samples):
        cum_values = np.zeros(x_vals.shape)
        y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
        cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
        inv_cdf = interp1d(cum_values / np.max(cum_values), x_vals)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    def sample_times_energies(self, size="infer", return_totals=True, **kw):
        """Sample interaction times and neutrino energies at those times
        also sample recoil energies based on those neutrino energies and
        the atomic cross-section at that energies.
        :param size: if 'infer' uses the expected number of total counts from the interaction
                    if a single integer, uses the same for each flavor
                    can also be a list of expected counts for each flavor
                    [nue, nue_bar, nux, nux_bar]
        neutrino_energies & recoil_energies can be passed as kwargs
        :return_totals: `bool` if True, returns the Totals, else returns all dict
        :returns: sampled_times (in seconds!), sampled_neutrino_energies, sampled_recoils
        """
        if self.snax_interactions is None:
            raise ValueError("No SNAX model is given")

        time_samples = dict()
        neutrino_energy_samples = dict()
        recoil_energy_samples = dict()
        # check how many interactions to sample from each flavor
        if type(size) == str:
            size = []
            for f in Flavor:
                tot_count = np.trapz(
                    self.snax_interactions.rates_per_recoil_scaled[f],
                    self.snax_interactions.recoil_energies,
                )
                size.append(int(tot_count.value))
        else:
            if np.ndim(size) == 0:
                size = np.repeat(size, 4)

        # for each flavor sample the times and energies
        for f, s in zip(Flavor, size):
            _time, _nu_energy, _recoil_energy = self._sample_times_energy(
                self.snax_interactions, s, flavor=f, **kw
            )
            time_samples[f] = _time
            neutrino_energy_samples[f] = _nu_energy
            recoil_energy_samples[f] = _recoil_energy

        time_samples["Total"] = np.concatenate([time_samples[f] for f in Flavor])
        neutrino_energy_samples["Total"] = np.concatenate(
            [neutrino_energy_samples[f] for f in Flavor]
        )
        recoil_energy_samples["Total"] = np.concatenate(
            [recoil_energy_samples[f] for f in Flavor]
        )
        if return_totals:
            return (
                time_samples["Total"],
                neutrino_energy_samples["Total"],
                recoil_energy_samples["Total"],
            )
        return time_samples, neutrino_energy_samples, recoil_energy_samples

    def _sample_times_energy(self, interaction, size, flavor=Flavor.NU_E, **kw):
        """
        For the sampling, I could in principle, sample an isotope for each interaction
        based on their abundance. However, this would slow down the sampling heavily,
        therefore, I select always the isotope that has the highest abundance
        To get a progress bar pass a leave=True kwarg
        """
        # fetch the attributes
        Model = interaction.Model
        times = Model.times.value  # in seconds
        neutrino_energies = (
            kw.get("neutrino_energies", None)
            or self.snax_interactions.Model.neutrino_energies.value
        )
        recoil_energies = (
            kw.get("recoil_energies", None)
            or self.snax_interactions.recoil_energies.value
        )
        leave = kw.get("leave", False)
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
            fluxes_at_times[i, :] = Model.model.get_initial_spectra(
                t=j * u.s, E=neutrino_energies * u.MeV, flavors=[flavor]
            )[flavor]
        # get all the cross-sections for a range of neutrino energies
        crosssec = self.snax_interactions.Nucleus[0].nN_cross_section(
            neutrino_energies * u.MeV, recoil_energies * u.keV
        )

        # calculate fluxes convolved with this cross-section
        flux_xsec = np.zeros(
            (len(sampled_times), len(recoil_energies), len(neutrino_energies))
        )
        for i, t in enumerate(sampled_times):
            flux_xsec[i] = (fluxes_at_times[i, :] * crosssec) / np.sum(
                fluxes_at_times[i, :] * crosssec
            )

        # select the most abundant atom
        maxabund = np.argmax(
            [nuc.abund for _, nuc in enumerate(self.snax_interactions.Nucleus)]
        )
        atom = self.snax_interactions.Nucleus[maxabund]

        sampled_nues, sampled_recoils = np.zeros(len(sampled_times)), np.zeros(
            len(sampled_times)
        )
        if not leave:
            pbar, length = enumerate(sampled_times), len(sampled_times)
        else:
            pbar, length = (
                tqdm(
                    enumerate(sampled_times), total=len(sampled_times), desc=flavor.name
                ),
                None,
            )

        for i, t in pbar:
            if length is not None:
                if i / length in [length * 0.25, length * 0.5, length * 0.75]:
                    print(f"\t{i / length:.2%} of {flavor.name} done")

            bb = np.trapz(flux_xsec[i], axis=0)
            sampled_nues[i] = self._inverse_transform_sampling(neutrino_energies, bb, 1)
            recspec = atom.nN_cross_section(
                sampled_nues[i] * u.MeV, recoil_energies * u.keV
            ).value.flatten()
            sampled_recoils[i] = self._inverse_transform_sampling(
                recoil_energies, recspec / np.sum(recspec), 1
            )[0]
        return sampled_times, sampled_nues, sampled_recoils


class SimulateSignal(SimulationInstructions):
    """Use either fuse or wfsim to simulate the signal
    It allows for simulating single or multiple supernova signals
    """

    def __init__(self, snax_interactions, instruction_type="fuse_microphysics"):
        super().__init__(snax_interactions)
        # self.Interaction = snax_interactions
        self.Model = snax_interactions.Model
        self.sim_folder = self.Model.config["wfsim"]["sim_folder"]
        self.csv_folder = self.Model.config["wfsim"]["instruction_path"]
        # self.instruction_generator = MultiSupernovaSimulations(snax_interactions)
        self.instruction_type = instruction_type
        self.model_hash = self.snax_interactions.Model.model_hash

    # def simulate_single(self, run_number=None, instructions=None, context=None, instruction_type=None, force=False, _multi=False):
    #     """Simulate the signal using the microphysics model
    #     :param run_number: optional, if None, fetches next available number for given hash
    #     :param instructions: `df` generated instructions (self.instruction_type or param instruction_type should match!)
    #     :param context: fuse/wfsim context, if None uses default (see self.fetch_context(None, "fuse"))
    #     :param instruction_type: `str` either "fuse_microphysics", "fuse_detectorphysics", "wfsim", if None, uses self.instruction_type
    #     :param force: `bool`, simulate even if exists
    #     Returns: Simulation context
    #     """
    #     type_of_instruction = instruction_type or self.instruction_type
    #     # get the context
    #     simulator = "fuse" if "fuse" in type_of_instruction else "wfsim"
    #     st = self.fetch_context(context, simulator)
    #     # check if the run number already exists
    #     run_number, isdone = self.get_run_number(run_number, st, type_of_instruction, is_multi=_multi)
    #     if isdone and not force:
    #         return st, run_number
    #
    #     # generate and save the instructions
    #     if instructions is None:
    #         if type_of_instruction == "fuse_microphysics":
    #             instructions = self.generate_fuse_microphysics_instructions()
    #         elif type_of_instruction == "fuse_detectorphysics":
    #             instructions = self.generate_fuse_detectorphysics_instructions()
    #         elif type_of_instruction == "wfsim":
    #             instructions = self.generate_wfsim_instructions_from_fuse(run_number)
    #         else:
    #             raise ValueError(
    #                 f"Instruction type {type_of_instruction} not recognized"
    #             )
    #
    #     csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
    #     instructions.to_csv(f"{self.csv_folder}/{csv_name}", index=False)
    #
    #     # make the simulation
    #     if type_of_instruction == "fuse_microphysics":
    #         st.set_config(
    #             {
    #                 "path": self.csv_folder,
    #                 "file_name": csv_name,
    #                 "n_interactions_per_chunk": 250,
    #                 "source_rate": 0,
    #             }
    #         )
    #         st.make(run_number, "microphysics_summary")
    #     elif type_of_instruction == "fuse_detectorphysics":
    #         import fuse
    #         # ChunkCsvInput needs to be registered
    #         st.register(fuse.detector_physics.ChunkCsvInput)
    #         st.set_config(
    #             {
    #                 "input_file": f"{self.csv_folder}/{csv_name}",
    #                 "n_interactions_per_chunk": 50,
    #                 "source_rate":0,
    #             }
    #         )
    #         st.make(run_number, "raw_records", progress_bar=True)
    #     elif type_of_instruction == "wfsim":
    #         st.set_config(dict(fax_file=f"{self.csv_folder}/{csv_name}"))
    #         st.make(run_number, "truth")
    #         st.make(run_number, "raw_records")
    #     else:
    #         raise ValueError(f"Instruction type {type_of_instruction} not recognized")
    #     print(f"Using {csv_name}\nSimulated run: {run_number}\nFor {type_of_instruction}")
    #     return st, run_number

    def simulate_single(self, run_number=None, instructions=None, context=None, instruction_type=None, force=False, _multi=False):
        """Simulate the signal using the microphysics model
        :param run_number: optional, if None, fetches next available number for given hash
        :param instructions: `df` generated instructions (self.instruction_type or param instruction_type should match!)
        :param context: fuse/wfsim context, if None uses default (see self.fetch_context(None, "fuse"))
        :param instruction_type: `str` either "fuse_microphysics", "fuse_detectorphysics", "wfsim", if None, uses self.instruction_type
        :param force: `bool`, simulate even if exists
        Returns: Simulation context
        """
        type_of_instruction = instruction_type or self.instruction_type
        # get the context
        simulator = "fuse" if "fuse" in type_of_instruction else "wfsim"
        st = self.fetch_context(context, simulator)
        # check if the run number already exists (also returns modified context)
        run_number, isdone, st = self.get_run_number(run_number, st, type_of_instruction, is_multi=_multi)
        if isdone and not force:
            return st, run_number

        csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
        full_path = os.path.join(self.csv_folder, csv_name)
        # generate and save the instructions
        if instructions is None:
            if type_of_instruction == "fuse_microphysics":
                instructions = self.generate_fuse_microphysics_instructions()
            elif type_of_instruction == "fuse_detectorphysics":
                instructions = self.generate_fuse_detectorphysics_instructions()
            elif type_of_instruction == "wfsim":
                instructions = self.generate_wfsim_instructions_from_fuse(run_number)
            else:
                raise ValueError(
                    f"Instruction type {type_of_instruction} not recognized"
                )

        instructions.to_csv(f"{full_path}", index=False)
        if type_of_instruction == "fuse_microphysics":
            st.make(run_number, "microphysics_summary")
        elif type_of_instruction == "fuse_detectorphysics":
            st.make(run_number, "raw_records", progress_bar=True)
        elif type_of_instruction == "wfsim":
            st.make(run_number, "truth")
            st.make(run_number, "raw_records")
        else:
            raise ValueError(
                f"Instruction type {type_of_instruction} not recognized"
            )
        print(f"Using {csv_name}\nSimulated run: {run_number}\nFor {type_of_instruction}")
        return st, run_number

    def simulate_multiple(
        self,
        run_number=None,
        number_of_supernova=None,
        rate=None,
        size=None,
        time_spacing_in_minutes=None,
        context=None,
        instruction_type=None,
    ):
        """Simulate multiple supernova signals
        If (number of supernova, time spacing) is provided, simulate that many signals,
        properly spaced in time and using actual model times
        else if (rate, size) is provided, simulate the signals with that rate and size
        ignore the model times and simulate a uniform distribution of signals in time
        Return: simulation context
        """
        is_time_independent = (rate is not None) and (size is not None)
        is_time_dependent = (number_of_supernova is not None) and (time_spacing_in_minutes is not None)
        type_of_instruction = instruction_type or self.instruction_type

        if not is_time_dependent and not is_time_independent:
            raise ValueError(
                "Please provide either number_of_supernova and time_spacing_in_minutes or rate and size"
            )

        if is_time_independent:
            instructions = self.uniform_time_instructions(
                rate, size, type_of_instruction
            )
        else:
            instructions = self.spaced_time_instructions(
                number_of_supernova= number_of_supernova,
                time_spacing_in_minutes=time_spacing_in_minutes,
                run_number=run_number,
                instruction_type=type_of_instruction
            )

        return self.simulate_single(run_number, instructions=instructions, context=context, _multi=True)

    def fetch_context(self, context=None, simulator="fuse"):
        """Fetch the context for the simulation
        If context is updated, change it in here
        Requires config to be a configparser object with ['wfsim']['sim_folder'] field
        So that the strax data folder can be found
        """
        data_folder = "fuse_data" if simulator == "fuse" else "strax_data"
        mc_data_folder = os.path.join(self.sim_folder, data_folder)

        def _add_strax_directory(context):
            # add the mc folder and return it
            output_folder_exists = False
            # check if the mc folder is already in the context
            for i, stores in enumerate(context.storage):
                if mc_data_folder in stores.path:
                    output_folder_exists = True
            # if it is not yet in the context, add a DataDirectory
            if not output_folder_exists:
                if not STRAXEXIST:
                    raise ImportError("strax not installed")
                import strax

                context.storage += [
                    strax.DataDirectory(mc_data_folder, readonly=False)
                ]

        # if a context is given, check if the storage is correct
        if context is not None:
            _add_strax_directory(context)
            return context
        # if no context is given, create a new one
        if simulator == "wfsim":
            if not WFSIMEXIST or not CUTAXEXIST:
                raise ImportError("wfsim or cutax not installed")
            import cutax

            context = cutax.contexts.xenonnt_sim_SR0v4_cmt_v9(
                output_folder=mc_data_folder
            )
        elif simulator == "fuse":
            if not CUTAXEXIST or not FUSEEXIST:
                raise ImportError("cutax or fuse not installed")
            import fuse, cutax
            from cutax.cut_lists.basic import BasicCuts

            # this cutax context has the proper detector conditions
            # do this if you are using the fuse_context branch (21.02.2024, it'll be merged soon)
            # context = cutax.contexts.xenonnt_fuse_full_chain_simulation(output_folder=mc_data_folder)

            # otherwise do this
            context = fuse.context.full_chain_context(output_folder=mc_data_folder,
                corrections_version = DEFAULT_XEDOCS_VERSION,
                simulation_config_file = DEFAULT_SIMULATION_VERSION,
                corrections_run_id = "026000",)
        else:
            raise ValueError(f"Simulator {simulator} not recognized")
        # add the strax folder to the context
        _add_strax_directory(context)
        return context

    # def get_run_number(self, run_number, st, instruction_type, is_multi=False):
    #     """ For simulations, check the hash and check existing simulated data
    #         Assign a new runid for each simulation
    #     """
    #     if instruction_type not in ["fuse_microphysics", "fuse_detectorphysics", "wfsim"]:
    #         raise ValueError(f"{instruction_type} is not recognized")
    #
    #     # unchanged, default context
    #     simulator = "fuse" if "fuse" in instruction_type else "wfsim"
    #     st = self.fetch_context(st, simulator)
    #
    #     exists = False
    #     # if not None check if data exists
    #     if run_number is not None:
    #         csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
    #         if instruction_type=="fuse_microphysics":
    #             st.set_config(
    #                 {
    #                     "path": self.csv_folder,
    #                     "file_name": csv_name,
    #                     "n_interactions_per_chunk": 250,
    #                     "source_rate": 0,
    #                 }
    #             )
    #             if st.is_stored(run_number, "microphysics_summary"):
    #                 print(f"microphysics summary exists for run number: {run_number}")
    #                 exists = True
    #         elif instruction_type=="fuse_detectorphysics":
    #             import fuse
    #             # ChunkCsvInput needs to be registered
    #             st.register(fuse.detector_physics.ChunkCsvInput)
    #             st.set_config(
    #                 {
    #                     "input_file": f"{self.csv_folder}/{csv_name}",
    #                     "n_interactions_per_chunk": 50,
    #                     "source_rate": 0,
    #                 }
    #             )
    #             exists = True
    #
    #         else:
    #             st.set_config(dict(fax_file=f"{self.csv_folder}/{csv_name}"))
    #             if st.is_stored(run_number, "raw_records"):
    #                 print(f"raw_records exists for run number: {run_number}")
    #                 exists = True
    #         # return run number and exist
    #         return run_number, exists
    #
    #     # if the run number is not passed, make one
    #     snewpyhash = self.Model.snewpy_hash
    #     if is_multi:
    #         snewpyhash = "multi_"+snewpyhash
    #
    #     count = 0
    #     while True:
    #         run_number = snewpyhash + f"_{count:05d}"
    #         csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
    #
    #         if instruction_type=="fuse_microphysics":
    #             st.set_config(
    #                 {
    #                     "path": self.csv_folder,
    #                     "file_name": csv_name,
    #                     "n_interactions_per_chunk": 250,
    #                     "source_rate": 0,
    #                 }
    #             )
    #             if not st.is_stored(run_number, "microphysics_summary"):
    #                 print(f" > {run_number} is not stored!!! , target=microphysics_summary, context_hash={st._context_hash()}")
    #                 break
    #             else:
    #                 print(f" > {run_number} exists")
    #
    #         elif instruction_type=="fuse_detectorphysics":
    #             import fuse
    #             # ChunkCsvInput needs to be registered
    #             st.register(fuse.detector_physics.ChunkCsvInput)
    #             st.set_config(
    #                 {
    #                     "input_file": f"{self.csv_folder}/{csv_name}",
    #                     "n_interactions_per_chunk": 50,
    #                     "source_rate": 0,
    #                 }
    #             )
    #             if not st.is_stored(run_number, "raw_records"):
    #                 print(f" > {run_number} is not stored!!! , target=raw_records, context_hash={st._context_hash()}")
    #                 break
    #             else:
    #                 print(f" > {run_number} exists")
    #
    #         else:
    #             st.set_config(dict(fax_file=f"{self.csv_folder}/{csv_name}"))
    #             if not st.is_stored(run_number, "raw_records"):
    #                 print(f" > {run_number} is not stored!!! , target=raw_records, context_hash={st._context_hash()}")
    #                 break
    #             else:
    #                 print(f" > {run_number} exists")
    #         count += 1
    #     return run_number, exists

    def get_run_number(self, run_number, st, instruction_type, is_multi=False):
        """ For simulations, check the hash and check existing simulated data
            Assign a new runid for each simulation
            returns, (run_number, exists, context)
        """
        if instruction_type not in ["fuse_microphysics", "fuse_detectorphysics", "wfsim"]:
            raise ValueError(f"{instruction_type} is not recognized")

        simulator = "fuse" if "fuse" in instruction_type else "wfsim"
        st = self.fetch_context(st, simulator)

        if run_number is not None:
            csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
            full_path = os.path.join(self.csv_folder, csv_name)
            config = {
                "path": self.csv_folder,
                "file_name": csv_name,
                "n_interactions_per_chunk": 250 if instruction_type == "fuse_microphysics" else 50,
                "source_rate": 0
            }

            if instruction_type == "fuse_microphysics":
                st.set_config(config)
                exists = st.is_stored(run_number, "microphysics_summary")
            elif instruction_type == "fuse_detectorphysics":
                import fuse
                st.register(fuse.detector_physics.ChunkCsvInput)
                config["input_file"] = f"{full_path}"
                st.set_config(config)
                exists = st.is_stored(run_number, "raw_records")
            else:
                config["fax_file"] = f"{full_path}"
                st.set_config(config)
                exists = st.is_stored(run_number, "raw_records")

            return run_number, exists, st

        snewpyhash = self.Model.snewpy_hash
        if is_multi:
            snewpyhash = "multi_" + snewpyhash

        count = 0
        while True:
            run_number = f"{snewpyhash}_{count:05d}"
            csv_name = f"instructions_{self.model_hash}_{run_number}.csv"
            full_path = os.path.join(self.csv_folder, csv_name)

            config = {
                "path": self.csv_folder,
                "file_name": csv_name,
                "n_interactions_per_chunk": 250 if instruction_type == "fuse_microphysics" else 50,
                "source_rate": 0
            }

            if instruction_type == "fuse_microphysics":
                st.set_config(config)
            elif instruction_type == "fuse_detectorphysics":
                import fuse
                st.register(fuse.detector_physics.ChunkCsvInput)
                config["input_file"] = f"{full_path}"
                st.set_config(config)
            else:
                config["fax_file"] = f"{full_path}"
                st.set_config(config)

            if not st.is_stored(run_number,
                                "microphysics_summary" if instruction_type == "fuse_microphysics" else "raw_records"):
                print(
                    f" > {run_number} is not stored!!! , target={'microphysics_summary' if instruction_type == 'fuse_microphysics' else 'raw_records'}, context_hash={st._context_hash()}")
                break
            else:
                print(f" > {run_number} exists")

            count += 1

        return run_number, False, st

    def what_is_hash_for(self, target_hash):
        """ For a given hash, return the snewpy model parameters
        """
        all_parameters = pd.read_csv(allparams_csv_path)
        try:
            # Query the DataFrame based on the "hash" column
            result_row = all_parameters[all_parameters['hash'] == target_hash].iloc[0]

            # Extract non-None columns
            non_none_columns = result_row[result_row.notnull()].index

            # Create a new DataFrame with non-None columns
            result_df = all_parameters.loc[all_parameters['hash'] == target_hash, non_none_columns]

            return result_df
        except IndexError:
            # Handle the case where the hash is not found in the DataFrame
            print(f"No entry found for hash: {target_hash}")
            return pd.DataFrame()

    # def see_simulated_contexts(self, context=None, simulator='fuse'):
    #     """See which simulations were made with what contexts"""
    #     st = self.fetch_context(context, simulator)
    #     data_folder = "fuse_data" if simulator == "fuse" else "strax_data"
    #     mc_data_folder = os.path.join(self.sim_folder, data_folder)
    #     from glob import glob
    #     simdirs = glob(mc_data_folder + "/*/")
    #     if
    #     files = set([s.split("/")[-2] for s in simdirs if "-" in s])
    #     hashes = np.array([h.split("-")[-1] for h in files])
    #     names = np.array([n.split("-")[0] for n in files])
    #     # unique hashes
    #     uh = np.unique([h for h in hashes if "temp" not in h])
    #     df_dict = {k: find_context_for_hash("truth", k) for k in uh}
    #     unames, uindex = np.unique(
    #         names, return_index=True
    #     )  # unique names and their indices
    #     uhashes = hashes[uindex]
    #     list_of_df = []
    #     for n, h in zip(unames, uhashes):
    #         h = h.split("_temp")[0]  # if there is a missing data
    #         df = df_dict[h].copy()  # copy the already-fetched dfs
    #         df["hash"] = [h] * len(df)  # some context e.g. dev points to more than one set
    #         df["sim_id"] = [n] * len(df)
    #         list_of_df.append(df)
    #     df_final = pd.concat(list_of_df)
    #     df_final["sn_model"] = df_final.apply(
    #         lambda row: "_".join(row["sim_id"].split("_")[:2]), axis=1
    #     )
    #     df_final.sort_values(by=["date_added", "sim_id"], inplace=True)
    #     df_final.reset_index(inplace=True)
    #     df_final.drop(columns="index", inplace=True)
    #     if unique:
    #         df_final.drop_duplicates(
    #             subset=["name", "tag", "hash", "sim_id", "sn_model"],
    #             keep="last",
    #             inplace=True,
    #         )
    #     if sim_id is not None:
    #         return df_final[df_final["sim_id"] == sim_id]
    #     df_final.reset_index(inplace=True)
    #     return df_final
