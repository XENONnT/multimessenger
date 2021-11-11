"""
Author: Melih Kara kara@kit.edu

The auxiliary tools that are used within the SN signal generation, waveform simulation

"""
from .libraries import *

def isnotebook():
    """ Tell if the script is running on a notebook

    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter

def _inverse_transform_sampling(x_vals, y_vals, n_samples):
    """ Inverse transform sampling

        Parameters
        ---------
        x_vals, y_vals : `array-like`
            The arrays to sample from
        n_samples : `int`
            The number of sample

    """
    n_samples = int(n_samples)
    cum_values = np.zeros(x_vals.shape)
    y_mid = (y_vals[1:]+y_vals[:-1])*0.5
    cum_values[1:] = np.cumsum(y_mid*np.diff(x_vals))
    inv_cdf = itp.interp1d(cum_values/np.max(cum_values), x_vals)
    r = np.random.rand(n_samples)
    return inv_cdf(r)

def get_rates_above_threshold(y_vals, rec_bins):
    """ Gives the total number of interactions above every given threshold
        The input is expected to be integrated rates over time. Thus, the
        total rate returned is counts/kg/t_full

        Parameters
        ----------
        y_vals : `array`
            The recoil rates
        rec_bins : `array`
            The sampling

        Returns
        -------
            The total number above each recoil energy, and corresponding recoil energies

    """
    rate_above_E = np.array([np.trapz(y_vals[i:], rec_bins[i:]) for i in range(len(rec_bins))])
    return rate_above_E, rec_bins #[:-1]

def interpolate_recoil_energy_spectrum(y_vals, rec_bins):
    """ Interpolate the coarsely sampled recoil energies

        Parameters
        ----------
        y_vals : `array`
            The recoil rates
        rec_bins : `array`
            The sampling

    """
    interpolated = itp.interp1d(rec_bins, y_vals, kind="cubic", fill_value="extrapolate")
    return interpolated

def sample_from_recoil_spectrum(x='energy', N_sample=1, pickled_file=None):
    """ Sample from the recoil spectrum in given file

        Parameters
        ----------
        x : `str`
            The x-axis to sample 'time' | 'energy'
        N_sample : `int`
            The number of samples
        pickled_file : `str`
            Path to pickled file. Expected to have (rates_Er, rates_t, recoil_e, timebins)

    """
    pickled_file = pickled_file or paths['data']+"rates_combined.pickle"
    with open(pickled_file, "rb") as input_file:
        rates_Er, rates_t, recoil_energy_bins, timebins  = pickle.load(input_file)
    if x.lower()=='energy':
        spectrum = rates_Er['Total']
        xaxis = recoil_energy_bins
        ## interpolate
        intrp_rates = itp.interp1d(xaxis, spectrum, kind="cubic", fill_value="extrapolate")
        xaxis = np.linspace(xaxis.min(), xaxis.max(), 200)
        spectrum = intrp_rates(xaxis)
    elif x.lower()=='time':
        spectrum = rates_t['Total']
        xaxis = timebins
    else: return print('choose x=time or x=energy')
    sample = _inverse_transform_sampling(xaxis, spectrum, N_sample)
    return sample

def instructions_SN(nevents_total, nevent_SN, single=False, dump_csv = False, filename = None, below_cathode = False):
    """
        WFSim instructions to simulate Supernova NR peak.

        Parameters
        ----------
        nevents_total : `int`
            total number of events desired
        nevent_SN: `int`
            The number of events/kg/(total duration) expected from a single SN.
            See `sn_utils.get_rates_above_threshold()`
        single : `bool`
            True if a single SN signal is desired. In which case the `nevents_total` is
            overwritten with the total nr of events *in the TPC* (not the nevent_SN)
        dump_csv : `bool`
            Whether to dump the csv file in config['wfsim']['instruction_path']
        filename : `str`
            The name of the csv file if the dump_csv is True
        below_cathode : `bool`
            If True, it randomly samples x positions 12 cm beyond cathode

        Notes
        -----
        - The distance between cathode and bottom array assumed to be 12 cm for now.
        - The number of total events is calculated by nevent_SN * volume in kg
        - For multiple SN events, the times are shifted by ten seconds to avoid overfilling the chunks
    """
    A, Z = 131.293, 54
    lxe_density = float(config['xenonnt']['lxe_density'])  # g/cm^3
    drift_field = float(config['xenonnt']['drift_field'])  # V/cm

    # compute the total expected interactions
    volume = float(config['xenonnt']['volume'])
    neventSN_total = nevent_SN*volume  # total nr SN events in the whole volume, after SN duration
    neventSN_total = np.ceil(neventSN_total).astype(int)

    # if single signal is requested, overwrite the total
    if single: nevents_total=neventSN_total

    # to get nevents_total number of events. We need to sample neventSN_total events for X times
    nr_iterations = np.ceil(nevents_total/ neventSN_total).astype(int)
    rolled_sample_size = int(nr_iterations * neventSN_total)

    # we need to sample this many energies and times
    sample_E = np.ones(rolled_sample_size) * - 1
    sample_t = np.ones(rolled_sample_size) * - 1

    ## shifted time sampling
    for i in range(nr_iterations):
        from_ = int(i * neventSN_total)
        to_ = int((i + 1) * neventSN_total)
        smpl_e = sample_from_recoil_spectrum(N_sample=neventSN_total)
        smpl_t = sample_from_recoil_spectrum(x='time', N_sample=neventSN_total)
        mint, maxt = np.min(sample_t), np.max(sample_t)
        # SN signal also has pre-SN neutrino, so if there are negative times boost them
        if mint <= 0: smpl_t -= mint

        time_shift = i * 10  # add 10 sec to each iteration
        sample_E[from_:to_] = smpl_e
        sample_t[from_:to_] = smpl_t + time_shift

    n = rolled_sample_size
    instructions = np.ones(2 * n, dtype=wfsim.instruction_dtype)
    instructions[:] = -1
    instructions['time'] = (1e8 * sample_t).repeat(2) + 1000000

    instructions['event_number'] = np.arange(0, n).repeat(2)
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'][:] = 0
    instructions['local_field'][:] = drift_field

    r = np.sqrt(np.random.uniform(0, straxen.tpc_r ** 2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    if below_cathode:
        instructions['z'] = np.repeat(np.random.uniform(-straxen.tpc_z - 12, 0, n), 2)
    else:
        instructions['z'] = np.repeat(np.random.uniform(-straxen.tpc_z, 0, n), 2)

    interaction_type = nestpy.INTERACTION_TYPE(0) # 0 for NR
    nc = nestpy.nestpy.NESTcalc(nestpy.nestpy.VDetector())

    quanta, exciton, recoil, e_dep = [], [], [], []
    for energy_deposit in tqdm(sample_E, desc='generating instructions from nest'):
        interaction = nestpy.INTERACTION_TYPE(interaction_type)
        y = nc.GetYields(interaction, energy_deposit, lxe_density, drift_field, A, Z)
        q = nc.GetQuanta(y, lxe_density)
        quanta.append(q.photons)
        quanta.append(q.electrons)
        exciton.append(q.excitons)
        exciton.append(0)
        # both S1 and S2
        recoil += [interaction_type, interaction_type]
        e_dep += [energy_deposit, energy_deposit]

    instructions['amp'] = quanta
    instructions['local_field'] = drift_field
    instructions['n_excitons'] = exciton
    instructions['recoil'] = recoil
    instructions['e_dep'] = e_dep
    instructions_df = pd.DataFrame(instructions)
    instructions_df = instructions_df[instructions_df['amp'] > 0]
    instructions_df.sort_values('time', inplace=True)
    if dump_csv:
        tdy = str(datetime.date.today())
        inst_path = config['wfsim']['instruction_path']
        filename = filename or f'{tdy}_instructions.csv'
        instructions_df.to_csv(f'{inst_path}{filename}', index=False)
        print(f'Saved in -> {inst_path}{filename}')
    return instructions_df


def clean_repos(pattern='*'):
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs_path']
    strax_data_path = config['wfsim']['strax_data_path']

    if input('Are you sure to delete all the data?\n'
            f'\t{inst_path}{pattern}\n'
            f'\t{logs_path}{pattern}\n'
            f'\t{strax_data_path}{pattern}\n>>>').lower() == 'y':
        os.system(f'rm -r {inst_path}{pattern}')
        os.system(f'rm -r {logs_path}{pattern}')
        os.system(f'rm -r {strax_data_path}{pattern}')


def see_repos():
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs_path']
    strax_data_path = config['wfsim']['strax_data_path']

    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    if not os.path.isdir(strax_data_path):
        os.mkdir(strax_data_path)
    click.secho('\n >>Instructions\n', bg='blue', color='white')
    os.system(f'ls -r {inst_path}')
    click.secho('\n >>Logs\n', bg='blue', color='white')
    os.system(f'ls -r {logs_path}')
    click.secho('\n >>Existing data\n', bg='blue', color='white')
    os.system(f'ls -r {strax_data_path}')
    
def display_config():
    for x in config.sections():
        click.secho(f'{x:^20}', bg='blue')
        for i in config[x]:
            print(i)
        print('-'*15)
    
def display_times(arr):
    """ Takes times array in ns, prints the corrected times
        and the duration

    """
    ti = int(arr.min()/1e9)
    tf = int(arr.max()/1e9)
    print(ti, datetime.datetime.utcfromtimestamp(ti).strftime('%Y-%m-%d %H:%M:%S'))
    print(tf, datetime.datetime.utcfromtimestamp(tf).strftime('%Y-%m-%d %H:%M:%S'))
    timedelta = datetime.datetime.utcfromtimestamp(tf)-datetime.datetime.utcfromtimestamp(ti)
    print(f'{timedelta.seconds} seconds \n{timedelta.resolution} resolution')


def inject_in(small_signal, big_signal):
    """ Inject small signal in a random time index of the big signal

    """
    # bring the small signal to zero
    small_signal['time'] -= small_signal['time'].min()
    # push it inside the big signal
    small_signal['time'] += np.random.choice(big_signal['time'])
    # check if it overlaps
    for time in small_signal['time']:
        if np.isclose(time, any(big_signal['time']), rtol=1e-8):
            print('Unlucky guess!')
            return inject_in(small_signal, big_signal)

    times_bkg = big_signal['time'].values
    times_sn = small_signal['time'].values
    # sanity check
    if (times_bkg.min() < times_sn.min()) & (times_bkg.max() > times_sn.max()):
        return small_signal
    else:
        click.secho('Something went wrong!', bg='red', bold=True)


def compute_rate_within(arr, sampling=1e9):
    """ Compute rates for any given sampling interval
        Allows to look for time intervals where the SN signal
        can be maximised

    """
    bins = np.arange(arr.min(), arr.max() + sampling, sampling)  # in seconds
    rates = np.histogram(arr, bins=bins)[0]
    return  bins, rates/sampling