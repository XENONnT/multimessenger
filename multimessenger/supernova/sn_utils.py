"""
Author: Melih Kara kara@kit.edu

The auxiliary tools that are used within the SN signal generation, waveform simulation

"""
import numpy as np
import scipy.interpolate as itp
import datetime
import os, click
import configparser
from scipy import interpolate
from glob import glob

# read in the configurations, by default it is the basic conf
# notice for wfsim related things, there is no field in the basic_conf
# config = configparser.ConfigParser()
# config.read('/dali/lgrandi/melih/mma/data/basic_conf.conf')

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


def add_strax_folder(config, context):
    """ This appends the SN MC folder to your directories
        So the simulations created by others are accessible to you

    """
    mc_folder = config["wfsim"]["sim_folder"]
    mc_data_folder = os.path.join(mc_folder, "strax_data")
    try:
        import strax
        st = context
        output_folder_exists = False
        for i, stores in enumerate(st.storage):
            if mc_data_folder in stores.path:
                output_folder_exists = True
        if not output_folder_exists:
            st.storage += [strax.DataDirectory(mc_data_folder, readonly=False)]
        return st
    except Exception as e:
        click.secho(f"> {e}", fg='red')
        pass

def clean_repos(pattern='*', config_file=None):
    config = configparser.ConfigParser()
    config_path = config_file or "../../simple_config.conf"
    #'/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_path)
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs']
    strax_data_path = config['wfsim']['sim_folder']

    for path in [inst_path, logs_path, strax_data_path]:
        if input('Are you sure to delete all the data?\n'
                 f'\t{path}{pattern}\n').lower() == 'y':
            os.system(f'rm -r {path}{pattern}')

def see_repos(config_file=None):
    config = configparser.ConfigParser()
    config_path = config_file or "../../simple_config.conf"
    #'/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_path)
    inst_path = config['wfsim']['instruction_path']
    logs_path = config['wfsim']['logs']
    strax_data_path = config['wfsim']['sim_folder']
    proc_data = config['paths']["processed_data"]

    for path in [inst_path, logs_path, strax_data_path, proc_data]:
        if not os.path.isdir(path):
            # os.mkdir(path)
            click.secho(f"> Could not found {path}", fg='red')
        else:
            click.secho(f'\n >> In {path}\n', bg='blue', fg='white')
            os.system(f'ls {path}*')

def see_simulated_files(config_file=None, get_names=False):
    """ Looks into the simulation folder and tells you
        the names of the simulated data
    """
    config = configparser.ConfigParser()
    config_path = config_file or "../../simple_config.conf"
    config.read(config_path)
    sim_folder = os.path.join(config['wfsim']['sim_folder'], "strax_data")
    simdirs = glob(sim_folder + '/*/')
    clean_simdirs = np.unique([a.split("-")[0].split("/")[-1] for a in simdirs])
    for i in clean_simdirs:
        print(f"\t{i}")
    if get_names:
        return clean_simdirs

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
    """ On dali, get the default config from mma repo
    """
    config = configparser.ConfigParser()
    config_file = config_file or '/dali/lgrandi/melih/mma/data/basic_conf.conf'
    config.read(config_file)
