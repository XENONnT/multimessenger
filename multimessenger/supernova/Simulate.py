# The core is taken from Andrii Terliuk's script
import numpy as np
import nestpy, os, click
from astropy import units as u
from snewpy.neutrino import Flavor
from scipy.interpolate import interp1d
from .sn_utils import isnotebook

if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
try:
    import wfsim
    WFSIMEXIST = True
except ImportError as e:
    WFSIMEXIST = False

try:
    import strax
    STRAXEXIST = True
except ImportError as e:
    STRAXEXIST = False

try:
    import cutax
    CUTAXEXIST = True
except ImportError as e:
    CUTAXEXIST = False


def generate_vertex(r_range=(0, 66.4),
                    z_range=(-148.15, 0), size=1):
    phi = np.random.uniform(size=size) * 2 * np.pi
    r = r_range[1] * np.sqrt(np.random.uniform((r_range[0] / r_range[1]) ** 2, 1, size=size))
    z = np.random.uniform(z_range[0], z_range[1], size=size)
    x = (r * np.cos(phi))
    y = (r * np.sin(phi))
    return x, y, z

def generate_times(rate, size, timemode='realistic'):
    # generating event times from exponential
    if timemode == "realistic":
        dt = np.random.exponential(1 / rate, size=size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return (times)
    elif timemode == "uniform":
        dt = (1 / rate) * np.ones(size - 1)
        times = np.append([1.0], 1.0 + dt.cumsum()) * 1e9
        times = times.round().astype(np.int64)
        return times

def generate_sn_instructions(energy_deposition,
                             n_tot=1000,
                             rate=20.,
                             fmap=None, field=None,
                             nc=None,
                             r_range=(0, 66.4), z_range=(-148.15, 0),
                             mode="all",
                             timemode="realistic",
                             time_offset=0,
                             **kwargs
                             ):
    if type(timemode) != str:
        times = timemode
    else:
        if timemode=="shifted":
            kwargs.pop("self")
            energy_deposition, times, n_tot = shifted_times(**kwargs)
        else:
            times = generate_times(rate=rate, size=n_tot, timemode=timemode) + time_offset
    if not WFSIMEXIST:
        raise ImportError("WFSim is not installed and is required for instructions!")
    instr = np.zeros(2 * n_tot, dtype=wfsim.instruction_dtype)
    instr['event_number'] = np.arange(1, n_tot + 1).repeat(2)
    instr['type'][:] = np.tile([1, 2], n_tot)
    instr['time'][:] = times.repeat(2)
    # generating uniformly distributed events for given R and Z range
    x, y, z = generate_vertex(r_range=r_range, z_range=z_range, size=n_tot)
    instr['x'][:] = x.repeat(2)
    instr['y'][:] = y.repeat(2)
    instr['z'][:] = z.repeat(2)
    # making energy
    instr['recoil'][:] = 7
    instr['e_dep'][:] = energy_deposition.repeat(2)
    # getting local field from field map
    if fmap is not None:
        instr['local_field'] = fmap(np.array([np.sqrt(x ** 2 + y ** 2), z]).T).repeat(2)
    else:
        if field is not None:
            instr['local_field'] = fmap
        else:
            raise TypeError('Provide a field, either a map or a single value')
    if nc is None:
        raise KeyError("You need to provide a nest instance")
    # And generating quanta from nest
    for i in range(0, n_tot):
        y = nc.GetYields(
            interaction=nestpy.INTERACTION_TYPE(instr['recoil'][2 * i]),
            energy=instr['e_dep'][2 * i],
            drift_field=instr['local_field'][2 * i],
        )
        q_ = nc.GetQuanta(y)
        instr['amp'][2 * i] = q_.photons
        instr['amp'][2 * i + 1] = q_.electrons
        instr['n_excitons'][2 * i:2 * (i + 1)] = q_.excitons
    if mode == "s1":
        instr = instr[instr['type'] == 1]
    elif mode == "s2":
        instr = instr[instr['type'] == 2]
    elif mode == "all":
        pass
    else:
        raise RuntimeError("Unknown mode: ", mode)
    # to avoid zero size interactions
    instr = instr[instr['amp'] > 0]
    return instr

def shifted_times(recoil_energies, times, rates_per_Er, rates_per_t, total, rate_in_oneSN):
    from .sn_utils import _inverse_transform_sampling
    xaxis_er = recoil_energies.value
    yaxis_er = rates_per_Er.value
    xaxis_t = times.value
    yaxis_t = rates_per_t.value

    single_sn_duration = np.ptp(times.value)
    rate_in_oneSN = int(rate_in_oneSN.value)
    nr_iterations, remainder = divmod(total, rate_in_oneSN)
    nr_iterations, remainder = int(nr_iterations), np.floor(remainder).astype(int)

    rolled_total = int(nr_iterations * rate_in_oneSN)
    sampled_er = np.zeros(rolled_total)
    sampled_t = np.zeros(rolled_total, dtype=np.int64)
    for i in range(nr_iterations):
        _sampled_Er_local = _inverse_transform_sampling(xaxis_er, yaxis_er, rate_in_oneSN)
        # sample times for one
        _sampled_t_local = (_inverse_transform_sampling(xaxis_t, yaxis_t, rate_in_oneSN)+1) * 1e9
        mint, maxt = np.min(_sampled_t_local), np.max(_sampled_t_local)
        #     # SN signal also has pre-SN neutrino, so if there are negative times boost them
        if mint <= 0:
            _sampled_t_local -= mint

        _from = int(i * rate_in_oneSN)
        _to = int((i + 1) * rate_in_oneSN)
        time_shift = i * single_sn_duration * 1e9
        sampled_t[_from:_to]  = _sampled_t_local + time_shift
        sampled_er[_from:_to] = _sampled_Er_local

    # add the remainder
    # Not needed for now. If a single SN has 30 evt, and 100 requested. Just simulate 90 for 3 SNs.
    # te remaining 10 will just confuse more.
    return sampled_er, sampled_t, rolled_total

def _simulate_one(df, runid, config, context):
    if not (WFSIMEXIST and CUTAXEXIST and STRAXEXIST):
        raise ImportError("WFSim, strax and/or cutax are not installed and is required for simulation!")
    csv_folder = config["wfsim"]["instruction_path"]
    csv_path = os.path.join(csv_folder, runid + ".csv")
    if not context.is_stored(runid, "truth"):
        df.to_csv(csv_path, index=False)
        context.set_config(dict(fax_file=csv_path))
        context.make(runid, "truth")
        # context.make(runid, "peak_basics")
        click.secho(f"{runid} is created! Returning context!", fg='blue')
    else:
        click.secho(f"{runid} already exists!", fg='green')
        if not os.path.isfile(csv_path):
            click.secho(f"{runid} exists in straxen storage, but the {csv_path} does not!"
                        f" Maybe manually deleted? Returning the simulation anyway", fg='red')
        context.make(runid, "truth")
        click.secho(f"{runid} is fetched! Returning context!", fg='green')
    return context

def _inverse_transform_sampling(x_vals, y_vals, n_samples):
    cum_values = np.zeros(x_vals.shape)
    y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
    cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
    inv_cdf = interp1d(cum_values / np.max(cum_values), x_vals)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


NEUTRINO_ENERGIES = np.linspace(0,250,500)
RECOIL_ENERGIES = np.linspace(0,30,100)

def _sample_times_energy(interaction, size, flavor=Flavor.NU_E, **kw):
    """
    For the sampling, I could in principle, sample an isotope for each interaction
    based on their abundance. However, this would slow down the sampling heavily,
    therefore, I select always the isotope that has the highest abundance
    """
    # fetch the attributes
    Model = interaction.Model
    times = Model.times.value
    neutrino_energies =  kw.get("neutrino_energies", None) or interaction.Model.neutrino_energies.value
    recoil_energies = kw.get("recoil_energies", None) or interaction.recoil_energies.value
    totrates = interaction.rates_per_time_scaled[flavor]

    # sample times
    sampled_times = _inverse_transform_sampling(times, totrates, size)
    sampled_times = np.sort(sampled_times)

    # fluxes at those times
    fluxes_at_times = Model.model.get_initial_spectra(t=sampled_times * u.s,
                                                      E=neutrino_energies * u.MeV,
                                                      flavors=[flavor])[flavor]
    # get all the cross-sections for a range of neutrino energies
    crosssec = interaction.Nucleus[0].nN_cross_section(neutrino_energies * u.MeV,
                                                       recoil_energies * u.keV)

    # calculate fluxes convolved with this cross-section
    flux_xsec = np.zeros((len(sampled_times), len(recoil_energies), len(neutrino_energies)))
    for i, t in enumerate(sampled_times):
        flux_xsec[i] = (fluxes_at_times[i, :] * crosssec) / np.sum(fluxes_at_times[i, :] * crosssec)

    # select the most abundant atom
    maxabund = np.argmax([nuc.abund for _, nuc in enumerate(interaction.Nucleus)])
    atom = interaction.Nucleus[maxabund]

    sampled_nues, sampled_recoils = np.zeros(len(sampled_times)), np.zeros(len(sampled_times))
    for i, t in tqdm(enumerate(sampled_times), total=len(sampled_times), desc=flavor.name):
        bb = np.trapz(flux_xsec[i], axis=0)
        sampled_nues[i] = _inverse_transform_sampling(neutrino_energies, bb, 1)
        recspec = atom.nN_cross_section(sampled_nues[i] * u.MeV, recoil_energies * u.keV).value.flatten()
        sampled_recoils[i] = _inverse_transform_sampling(recoil_energies, recspec / np.sum(recspec), 1)[0]
    return sampled_times, sampled_nues, sampled_recoils



def sample_times_energies(interaction, size='infer', **kw):
    """ Sample interaction times and neutrino energies at those times
        also sample recoil energies based on those neutrino energies and
        the atomic cross-section at that energies.
        :param interaction: `interactions.Interactions object`
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
    if type(size)==str:
        size = []
        for f in Flavor:
            tot_count = np.trapz(interaction.rates_per_recoil_scaled[f], interaction.recoil_energies)
            size.append(int(tot_count.value))
    else:
        if np.ndim(size)==0:
            size = np.repeat(size, 4)

    for f,s in zip(Flavor, size):
        a,b,c = _sample_times_energy(interaction, s, flavor=f, **kw)
        time_samples[f] = a
        neutrino_energy_samples[f] = b
        recoil_energy_samples[f] = c

    time_samples['Total'] = np.concatenate([time_samples[f] for f in Flavor])
    neutrino_energy_samples['Total'] = np.concatenate([neutrino_energy_samples[f] for f in Flavor])
    recoil_energy_samples['Total'] = np.concatenate([recoil_energy_samples[f] for f in Flavor])

    return time_samples, neutrino_energy_samples, recoil_energy_samples


