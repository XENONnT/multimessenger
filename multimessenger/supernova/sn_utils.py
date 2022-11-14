"""
Author: Melih Kara kara@kit.edu

The auxiliary tools that are used within the SN signal generation, waveform simulation

"""
import numpy as np
import pandas as pd
import scipy.interpolate as itp
import _pickle as pickle
import datetime
import os, click
import configparser
from scipy import interpolate

# read in the configurations, by default it is the basic conf
# notice for wfsim related things, there is no field in the basic_conf
config = configparser.ConfigParser()
config.read('/dali/lgrandi/melih/mma/data/basic_conf.conf')

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
    cum_values = np.zeros(x_vals.shape)
    y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
    cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
    inv_cdf = interpolate.interp1d(cum_values / np.max(cum_values), x_vals)
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
    return rate_above_E, rec_bins


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


# def sample_from_recoil_spectrum(x='energy', N_sample=1, pickled_file=None, config_file=None):
#     """ Sample from the recoil spectrum in given file
#
#         Parameters
#         ----------
#         x : `str`
#             The x-axis to sample 'time' | 'energy'
#         N_sample : `int`
#             The number of samples
#         pickled_file : `str`
#             Path to pickled file. Expected to have (rates_Er, rates_t, recoil_e, timebins)
#         config_file : `str`
#             Path to configuration file
#
#     """
#     config = configparser.ConfigParser()
#     config_path = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
#     config.read(config_path)
#     path_img = config['paths']['imgs']
#     path_data = config['paths']['data']
#     paths = {'img': path_img, 'data': path_data}
#
#     pickled_file = pickled_file or paths['data']+"rates_combined.pickle"
#     with open(pickled_file, "rb") as input_file:
#         rates_Er, rates_t, recoil_energy_bins, timebins = pickle.load(input_file)
#     if x.lower() == 'energy':
#         spectrum = rates_Er['Total']
#         xaxis = recoil_energy_bins
#         # interpolate
#         intrp_rates = itp.interp1d(xaxis, spectrum, kind="cubic", fill_value="extrapolate")
#         xaxis = np.linspace(xaxis.min(), xaxis.max(), 200)
#         spectrum = intrp_rates(xaxis)
#     elif x.lower() == 'time':
#         spectrum = rates_t['Total']
#         xaxis = timebins
#     else: return print('choose x=time or x=energy')
#     sample = _inverse_transform_sampling(xaxis, spectrum, N_sample)
#     return sample


# def instructions_SN(total_events_to_sim, sn_perton_fullt, single_sn_duration=10, single=False,
#                     dump_csv=False, filename=None, below_cathode=False, config_file=None):
#     # Todo: to study the signal shape, times are not really relevant
#     # and the time sampling makes things hard, maybe we can ignore that and only sample energies
#     """
#         WFSim instructions to simulate Supernova NR peak.
#
#         Parameters
#         ----------
#         total_events_to_sim : `int`
#             total number of events desired
#         sn_perton_fullt: `float`
#             The number of events/tonne/(total duration) expected from a single SN.
#             See `sn_utils.get_rates_above_threshold()`
#         single_sn_duration: `float`
#             The "total duration" of a single supernova signal
#         single : `bool`
#             True if a single SN signal is desired. In which case the `total_events_to_sim` is
#             overwritten with the total nr of events *in the TPC* (not the nevent_SN)
#         dump_csv : `bool`
#             Whether to dump the csv file in config['wfsim']['instruction_path']
#         filename : `str`
#             The name of the csv file if the dump_csv is True
#         below_cathode : `bool`
#             If True, it randomly samples x positions 12 cm beyond cathode
#         config_file : `str`
#             Path to configuration file
#
#         Notes
#         -----
#         - The distance between cathode and bottom array assumed to be 12 cm for now.
#         - The number of total events is calculated by nevent_SN * volume in kg
#         - For multiple SN events, the times are shifted by ten seconds to avoid overfilling the chunks
#     """
#     import wfsim, nestpy, straxen
#     if isnotebook():
#         from tqdm.notebook import tqdm
#     else:
#         from tqdm import tqdm
#
#     config = configparser.ConfigParser()
#     config_path = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
#     config.read(config_path)
#
#     # Xenon Atom
#     A, Z = 131.293, 54
#     lxe_density = float(config['xenonnt']['lxe_density'])  # g/cm^3
#     drift_field = float(config['xenonnt']['drift_field'])  # V/cm
#
#     # compute the total expected interactions
#     volume = float(config['xenonnt']['volume'])
#     sn_pert = sn_perton_fullt * volume  # total nr SN events in the whole volume, after SN duration
#     sn_pert = np.ceil(sn_pert).astype(int)
#
#     # if single signal is requested, overwrite the total
#     if single:
#         total_events_to_sim = sn_pert
#
#     # to get total number of events. We need to sample sn_pert events for X times
#     nr_iterations = np.ceil(total_events_to_sim / sn_pert).astype(int)
#     rolled_sample_size = int(nr_iterations * sn_pert)
#
#     # we need to sample this many energies and times
#     sample_E = np.ones(rolled_sample_size) * - 1
#     sample_t = np.ones(rolled_sample_size) * - 1
#
#     ## shifted time sampling
#     for i in range(nr_iterations):
#         from_ = int(i * sn_pert)
#         to_ = int((i + 1) * sn_pert)
#         smpl_e = sample_from_recoil_spectrum(N_sample=sn_pert)
#         smpl_t = (sample_from_recoil_spectrum(x='time', N_sample=sn_pert) * 1e9).astype(np.int64) # nanosec
#         mint, maxt = np.min(smpl_t), np.max(smpl_t)
#         # SN signal also has pre-SN neutrino, so if there are negative times boost them
#         if mint <= 0: smpl_t -= mint
#
#         time_shift = i * single_sn_duration * 1e9 # add 1 SN duration to each iteration
#         sample_E[from_:to_] = smpl_e
#         sample_t[from_:to_] = smpl_t + time_shift
#
#     n = rolled_sample_size
#     instructions = np.ones(2 * n, dtype=wfsim.instruction_dtype)
#     instructions[:] = -1
#     instructions['time'] = (sample_t).repeat(2) + 1000000
#
#     instructions['event_number'] = np.arange(0, n).repeat(2)
#     instructions['type'] = np.tile([1, 2], n)
#     instructions['recoil'][:] = 0
#     instructions['local_field'][:] = drift_field
#
#     r = np.sqrt(np.random.uniform(0, straxen.tpc_r ** 2, n))
#     t = np.random.uniform(-np.pi, np.pi, n)
#     instructions['x'] = np.repeat(r * np.cos(t), 2)
#     instructions['y'] = np.repeat(r * np.sin(t), 2)
#
#     if below_cathode:
#         instructions['z'] = np.repeat(np.random.uniform(-straxen.tpc_z - 12, 0, n), 2)
#     else:
#         instructions['z'] = np.repeat(np.random.uniform(-straxen.tpc_z, 0, n), 2)
#
#     interaction_type = nestpy.INTERACTION_TYPE(0) # 0 for NR
#     nc = nestpy.nestpy.NESTcalc(nestpy.nestpy.VDetector())
#
#     quanta, exciton, recoil, e_dep = [], [], [], []
#     for energy_deposit in tqdm(sample_E, desc='generating instructions from nest'):
#         interaction = nestpy.INTERACTION_TYPE(interaction_type)
#         y = nc.GetYields(interaction, energy_deposit, lxe_density, drift_field, A, Z)
#         q = nc.GetQuanta(y, lxe_density)
#         quanta.append(q.photons)
#         quanta.append(q.electrons)
#         exciton.append(q.excitons)
#         exciton.append(0)
#         # both S1 and S2
#         recoil += [interaction_type, interaction_type]
#         e_dep += [energy_deposit, energy_deposit]
#
#     instructions['amp'] = quanta
#     instructions['local_field'] = drift_field
#     instructions['n_excitons'] = exciton
#     instructions['recoil'] = recoil
#     instructions['e_dep'] = e_dep
#     instructions_df = pd.DataFrame(instructions)
#     instructions_df = instructions_df[instructions_df['amp'] > 0]
#     instructions_df.sort_values('time', inplace=True)
#     if dump_csv:
#         tdy = str(datetime.date.today())
#         inst_path = config['wfsim']['instruction_path']
#         filename = filename or f'{tdy}_instructions.csv'
#         instructions_df.to_csv(f'{inst_path}{filename}', index=False)
#         print(f'Saved in -> {inst_path}{filename}')
#     return instructions_df


def clean_repos(pattern='*', config_file=None):
    config = configparser.ConfigParser()
    config_path = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_path)
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs_path']
    strax_data_path = config['wfsim']['sim_folder']

    if input('Are you sure to delete all the data?\n'
             f'\t{inst_path}{pattern}\n'
             f'\t{logs_path}{pattern}\n'
             f'\t{strax_data_path}{pattern}\n>>>').lower() == 'y':
        os.system(f'rm -r {inst_path}{pattern}')
        os.system(f'rm -r {logs_path}{pattern}')
        os.system(f'rm -r {strax_data_path}{pattern}')


def see_repos(config_file=None):
    config = configparser.ConfigParser()
    config_path = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_path)
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs_path']
    strax_data_path = config['wfsim']['sim_folder']

    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
    if not os.path.isdir(strax_data_path):
        os.mkdir(strax_data_path)
    click.secho('\n >>Instructions\n', bg='blue', fg='white')
    os.system(f'ls -r {inst_path}')
    click.secho('\n >>Logs\n', bg='blue', fg='white')
    os.system(f'ls -r {logs_path}')
    click.secho('\n >>Existing data\n', bg='blue', fg='white')
    os.system(f'ls -r {strax_data_path}')


def display_config(config_file=None):
    config = configparser.ConfigParser()
    config_file = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_file)
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


def compute_rate_within(arr, sampling=1, ts=None, tf=None, start_at_zero=True, shift_time=None):
    """ Compute rates for any given sampling interval
        Allows to look for time intervals where the SN signal
        can be maximised

        :param arr: `array` times in nanosec
        :param sampling: `float` sampling size in sec
        :param ts, tf: `float` start, finish times in sec

    """
    arr = arr * 1e-9
    if start_at_zero:
        # start at zero
        arr -= np.min(arr)
    if type(shift_time) == type(None):
        pass
    elif type(shift_time) == str:
        # select a time within half an hour
        random_sec = np.random.choice(np.linspace(0, 1800))
        arr += random_sec
    elif type(shift_time) == float or type(shift_time) == int:
        arr += shift_time
    else:
        raise ValueError('shift time can either be a float, None or \'random\'')
    ts = ts or 0
    tf = tf or np.max(arr)
    bins = np.arange(ts, tf + sampling, sampling)  # in seconds
    return np.histogram(arr, bins=bins)

def get_config(config_file=None):
    config = configparser.ConfigParser()
    config_file = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_file)
