# The core is taken from Andrii Terliuk's script
import numpy as np
import nestpy, os, click

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
    instr['recoil'][:] = nestpy.INTERACTION_TYPE.NR
    instr['e_dep'][:] = energy_deposition.repeat(2)
    # getting local field from field map
    if fmap is not None:
        instr['local_field'] = fmap(np.array([np.sqrt(x ** 2 + y ** 2), z]).T).repeat(2)
    else:
        if field is not None:
            instr['local_field'] = field
        else:
            raise TypeError('Provide a field, either a map or a single value')

    # And generating quanta from nest
    nc = nestpy.NESTcalc(nestpy.VDetector())
    A = 131.293
    Z = 54.
    density = 2.862  # g/cm^3
    for i in range(0, n_tot):
        interaction = nestpy.INTERACTION_TYPE(instr['recoil'][2 * i])
        y = nc.GetYields(interaction=interaction,
                         energy=instr['e_dep'][2 * i],
                         density=density,
                         drift_field=instr['local_field'][2 * i],
                         A=A, Z=Z, )
        q_ = nc.GetQuanta(y, density)
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
