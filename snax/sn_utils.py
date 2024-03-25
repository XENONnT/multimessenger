"""
Author: Melih Kara kara@kit.edu

The auxiliary tools that are used within the SN signal generation, waveform simulation

"""

import click
import configparser
import datetime
import json
import os
from glob import glob
import numpy as np
import pandas as pd
import scipy.interpolate as itp
from scipy import interpolate
from base64 import b32encode
from hashlib import sha1
from collections.abc import Mapping


# read in the configurations, by default it is the basic conf
# notice for wfsim related things, there is no field in the basic_conf
# config = configparser.ConfigParser()
# config.read("/project2/lgrandi/xenonnt/simulations/supernova/simple_config.conf")


def isnotebook():
    """Tell if the script is running on a notebook"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def validate_config_file(config_file_path):
    # Define the expected sections and options
    expected_sections = ["paths", "wfsim"]
    expected_options = {
        "paths": ["base", "snewpy_models", "processed_data", "imgs", "data", "outputs"],
        "wfsim": ["sim_folder", "instruction_path", "logs"],
    }

    # Read the user-provided config file
    user_config = configparser.ConfigParser()
    user_config.read(config_file_path)

    # Check if all expected sections are present
    for section in expected_sections:
        if section not in user_config:
            return (
                False,
                f"Section '{section}' is missing in the config file.\n>{config_file_path}",
            )

    # Check if all expected options are present in each section
    for section, options in expected_options.items():
        for option in options:
            if option not in user_config[section]:
                return (
                    False,
                    f"Option '{option}' is missing in the '{section}' section in the config file."
                    f"\n>{config_file_path}",
                )

    # If all checks pass, the config file is valid
    return True, "Config file is valid."


def _inverse_transform_sampling(x_vals, y_vals, n_samples):
    cum_values = np.zeros(x_vals.shape)
    y_mid = (y_vals[1:] + y_vals[:-1]) * 0.5
    cum_values[1:] = np.cumsum(y_mid * np.diff(x_vals))
    inv_cdf = interpolate.interp1d(cum_values / np.max(cum_values), x_vals)
    r = np.random.rand(n_samples)
    return inv_cdf(r)


def get_rates_above_threshold(y_vals, rec_bins):
    """Gives the total number of interactions above every given threshold
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
    rate_above_E = np.array(
        [np.trapz(y_vals[i:], rec_bins[i:]) for i in range(len(rec_bins))]
    )
    return rate_above_E, rec_bins


def interpolate_recoil_energy_spectrum(y_vals, rec_bins):
    """Interpolate the coarsely sampled recoil energies

    Parameters
    ----------
    y_vals : `array`
        The recoil rates
    rec_bins : `array`
        The sampling

    """
    interpolated = itp.interp1d(
        rec_bins, y_vals, kind="cubic", fill_value="extrapolate"
    )
    return interpolated


# def fetch_context(config):
#     """If context is updated, change it in here
#     Requires config to be a configparser object with ['wfsim']['sim_folder'] field
#     So that the strax data folder can be found
#     """
#     mc_folder = config["wfsim"]["sim_folder"]
#     mc_data_folder = os.path.join(mc_folder, "strax_data")
#     import cutax
#
#     return cutax.contexts.xenonnt_sim_SR0v4_cmt_v9(output_folder=mc_data_folder)


def add_strax_folder(config, context):
    """This appends the SN MC folder to your directories
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
        click.secho(f"> {e}", fg="red")
        pass


def clean_repos(pattern="*", config_file=None):
    config = configparser.ConfigParser()
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
    )
    config_path = config_file or default_config_path
    config.read(config_path)
    inst_path = config["wfsim"]["instruction_path"]
    logs_path = config["wfsim"]["logs"]
    strax_data_path = config["wfsim"]["sim_folder"]

    for path in [inst_path, logs_path, strax_data_path]:
        if (
            input(
                "Are you sure to delete all the data?\n" f"\t{path}{pattern}\n"
            ).lower()
            == "y"
        ):
            os.system(f"rm -r {path}{pattern}")


def see_repos(config_file=None):
    config = configparser.ConfigParser()
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
    )
    config_path = config_file or default_config_path
    config.read(config_path)
    inst_path = config["wfsim"]["instruction_path"]
    logs_path = config["wfsim"]["logs"]
    strax_data_path = config["wfsim"]["sim_folder"]
    proc_data = config["paths"]["processed_data"]

    click.secho(
        f"\n >> In {strax_data_path} There are these folders\n", bg="green", fg="white"
    )
    for path in [inst_path, logs_path, proc_data]:
        if not os.path.isdir(path):
            # os.mkdir(path)
            click.secho(f"> Could not found {path}", fg="red")
        else:
            click.secho(f"\n >> In {path}\n", bg="blue", fg="white")
            os.system(f"ls {path}*/")


def see_simulated_files(config_file=None, get_names=False):
    """Looks into the simulation folder and tells you
    the names of the simulated data
    """
    config = configparser.ConfigParser()
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
    )
    config_path = config_file or default_config_path
    config.read(config_path)
    sim_folder = os.path.join(config["wfsim"]["sim_folder"], "strax_data")
    simdirs = glob(sim_folder + "/*/")
    clean_simdirs = np.unique([a.split("-")[0].split("/")[-1] for a in simdirs])
    if get_names:
        return clean_simdirs
    for i in clean_simdirs:
        print(f"\t{i}")


def display_config(config_file=None):
    config = configparser.ConfigParser()
    default_config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
    )
    config_path = config_file or default_config_path
    config.read(config_path)
    try:
        for x in config.sections():
            click.secho(f"{x:^20}", bg="blue")
            for i in config[x]:
                print(f"{i:20} : {config[x][i]}")
            print("-" * 15)
    except Exception as e:
        print(f"{e}\nSomething went wrong, maybe empty config?")


def display_times(arr):
    """Takes times array in ns, prints the corrected times
    and the duration

    """
    ti = int(arr.min() / 1e9)
    tf = int(arr.max() / 1e9)
    print(ti, datetime.datetime.utcfromtimestamp(ti).strftime("%Y-%m-%d %H:%M:%S"))
    print(tf, datetime.datetime.utcfromtimestamp(tf).strftime("%Y-%m-%d %H:%M:%S"))
    timedelta = datetime.datetime.utcfromtimestamp(
        tf
    ) - datetime.datetime.utcfromtimestamp(ti)
    print(f"{timedelta.seconds} seconds \n{timedelta.resolution} resolution")


def _see_all_contexts():
    import utilix

    entries = utilix.rundb.xent_collection(collection="contexts").find(
        projection={"name": True}
    )
    contexts = [i.get("name") for i in entries]
    contexts = np.unique(contexts)
    return contexts


def find_context_for_hash(
    data_type: str,
    lineage_hash: str,
    columns=(
        "name",
        "strax_version",
        "cutax_version",
        "straxen_version",
        "date_added",
        "tag",
    ),
):
    """Find back the software and context that was used"""
    import utilix

    # Query the context database for the requested datatype
    entries = utilix.rundb.xent_collection(collection="contexts").find(
        {f"hashes.{data_type}": lineage_hash}, projection={key: True for key in columns}
    )

    # Cast the docs into a format that allows making a dataframe
    df = pd.DataFrame([{key: doc.get(key) for key in columns} for doc in entries])
    return df


def see_simulated_contexts(config_file=None, sim_id=None, unique=True):
    """See which simulations were made with what contexts"""
    # check the lineages in the simulated files
    if config_file is None or type(config_file) == str:
        config_file = config_file or os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "..", "simple_config.conf"
        )
        config = configparser.ConfigParser()
        config.read(config_file)
    else:
        config = config_file

    sim_folder = os.path.join(config["wfsim"]["sim_folder"], "strax_data")
    simdirs = glob(sim_folder + "/*/")
    files = [s.split("/")[-2] for s in simdirs if "-truth-" in s]
    hashes = np.array([h.split("-")[-1] for h in files])
    names = np.array([n.split("-")[0] for n in files])
    # unique hashes
    uh = np.unique([h for h in hashes if "temp" not in h])
    df_dict = {k: find_context_for_hash("truth", k) for k in uh}
    unames, uindex = np.unique(
        names, return_index=True
    )  # unique names and their indices
    uhashes = hashes[uindex]
    list_of_df = []
    for n, h in zip(unames, uhashes):
        h = h.split("_temp")[0]  # if there is a missing data
        df = df_dict[h].copy()  # copy the already-fetched dfs
        df["hash"] = [h] * len(df)  # some context e.g. dev points to more than one set
        df["sim_id"] = [n] * len(df)
        list_of_df.append(df)
    df_final = pd.concat(list_of_df)
    df_final["sn_model"] = df_final.apply(
        lambda row: "_".join(row["sim_id"].split("_")[:2]), axis=1
    )
    df_final.sort_values(by=["date_added", "sim_id"], inplace=True)
    df_final.reset_index(inplace=True)
    df_final.drop(columns="index", inplace=True)
    if unique:
        df_final.drop_duplicates(
            subset=["name", "tag", "hash", "sim_id", "sn_model"],
            keep="last",
            inplace=True,
        )
    if sim_id is not None:
        return df_final[df_final["sim_id"] == sim_id]
    df_final.reset_index(inplace=True)
    return df_final


def check_stored(st, df, keys=None):
    """Check if the given keys for the given run_id are stored in context st
    by default checks for peak_basics and peak_positions
    """
    if keys is None:
        keys = ["peak_basics", "peak_positions"]
    if not isinstance(keys, list):
        keys = [keys]
    for k in keys:
        df[f"{k}_stored"] = df.apply(lambda row: st.is_stored(row["sim_id"], k), axis=1)
    return df


def inject_in(small_signal, big_signal):
    """Inject small signal in a random time index of the big signal"""
    # bring the small signal to zero
    small_signal["time"] -= small_signal["time"].min()
    # push it inside the big signal
    small_signal["time"] += np.random.choice(big_signal["time"])
    # check if it overlaps
    for time in small_signal["time"]:
        if np.isclose(time, any(big_signal["time"]), rtol=1e-8):
            print("Unlucky guess!")
            return inject_in(small_signal, big_signal)

    times_bkg = big_signal["time"].values
    times_sn = small_signal["time"].values
    # sanity check
    if (times_bkg.min() < times_sn.min()) & (times_bkg.max() > times_sn.max()):
        return small_signal
    else:
        click.secho("Something went wrong!", bg="red", bold=True)


def compute_rate_within(
    arr, sampling=1, ts=None, tf=None, start_at_zero=True, shift_time=None
):
    """Compute rates for any given sampling interval
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
        raise ValueError("shift time can either be a float, None or 'random'")
    ts = ts or 0
    tf = tf or np.max(arr)
    bins = np.arange(ts, tf + sampling, sampling)  # in seconds
    return np.histogram(arr, bins=bins)


def get_config(config_file=None):
    """On dali, get the default config from mma repo"""
    config = configparser.ConfigParser()
    config_file = (
        config_file
        or "/project2/lgrandi/xenonnt/simulations/supernova/simple_config.conf"
    )
    config.read(config_file)
    return config


def deterministic_hash(thing, length=10):
    """Return a base32 lowercase string of length determined from hashing a container hierarchy.
    from strax package
    """

    hashable = hashablize(thing)
    jsonned = json.dumps(hashable, cls=NumpyJSONEncoder)
    # disable bandit
    digest = sha1(jsonned.encode("ascii")).digest()
    return b32encode(digest)[:length].decode("ascii").lower()


class NumpyJSONEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    Edited from mpl3d: mpld3/_display.py
    from strax package
    """

    def default(self, obj):
        try:
            iterable = iter(obj)
        except TypeError:
            pass
        else:
            return [self.default(item) for item in iterable]
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def hashablize(obj):
    """Convert a container hierarchy into one that can be hashed.
    See http://stackoverflow.com/questions/985294
    from strax package

    """
    if isinstance(obj, Mapping):
        # Convert immutabledict etc for json decoding
        obj = dict(obj)
    try:
        hash(obj)
    except TypeError:
        if isinstance(obj, dict):
            return tuple((k, hashablize(v)) for (k, v) in sorted(obj.items()))
        elif isinstance(obj, np.ndarray):
            return tuple(obj.tolist())
        elif hasattr(obj, "__iter__"):
            return tuple(hashablize(o) for o in obj)
        else:
            raise TypeError("Can't hashablize object of type %r" % type(obj))
    else:
        return obj

def get_hash_from_model(initiated_model):
    """ Get the hash from imported snewpy model (initialized)
        Uses the metadata and the model name to get a generic hash
    """
    import astropy
    meta_items = initiated_model.metadata.items()
    _meta = {k: v.value if isinstance(v, astropy.units.quantity.Quantity) else v for k, v in meta_items}
    # _meta['model_name'] = model.filename # not all models have a file name
    _meta['model_name'] = initiated_model.__class__.__name__
    return deterministic_hash(_meta)


def what_is_hash_for(target_hash):
    """ For a given hash, return the snewpy model parameters
    """
    # all params file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(script_dir, os.pardir)
    allparams_csv_path = os.path.join(parent_dir, 'all_parameters.csv')
    all_parameters = pd.read_csv(allparams_csv_path)
    if "_" in target_hash:
        parts = target_hash.split('_')
        target_hash = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[0]

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


def fetch_context(context=None,
                  simulator="fuse",
                  instruction_type="fuse_microphysics",
                  mcdata_folder=None,
                  csv_folder="./",
                  corrections_version='global_v14'):
    """
    Fetch the context for the simulation
    If context is updated, change it in here
    Requires config to know mc data storage, and where the csv files are written
    So that the strax data folder can be found
    """
    # check if all modules exists
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
    CUTAXEXIST = module_exists.get("CUTAXEXIST", False)
    FUSEEXIST = module_exists.get("FUSEEXIST", False)

    def _add_strax_directory(context):
        # add the mc folder and return it
        output_folder_exists = False
        # check if the mc folder is already in the context
        for i, stores in enumerate(context.storage):
            if mcdata_folder in stores.path:
                output_folder_exists = True
        # if it is not yet in the context, add a DataDirectory
        if not output_folder_exists:
            if not STRAXEXIST:
                raise ImportError("strax not installed")
            import strax

            context.storage += [
                strax.DataDirectory(mcdata_folder, readonly=False)
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
            output_folder=mcdata_folder
        )
        # do we have to modify config here?

    elif simulator == "fuse":
        if not CUTAXEXIST or not FUSEEXIST:
            raise ImportError("cutax or fuse not installed")
        import fuse, cutax
        # make sure you are checked out at fuse 1.1.0 version!! (git checkout 1.1.0)
        context = fuse.context.full_chain_context(output_folder=mcdata_folder,
            corrections_version = corrections_version,
            # simulation_config_file = DEFAULT_SIMULATION_VERSION,
            corrections_run_id = "026000",)


        if instruction_type=="fuse_microphysics":
            config = {"path": csv_folder,
                      "n_interactions_per_chunk": 250,
                      "source_rate": 0 }
            context.set_config(config)
        elif instruction_type=="fuse_detectorphysics":
            context.register(fuse.detector_physics.ChunkCsvInput)
    else:
        raise ValueError(f"Simulator {simulator} not recognized")
    # add the strax folder to the context
    _add_strax_directory(context)
    return context

# def make_json(inter, sim_id, config_file, jsonfilename="simulation_metadata.json"):
#     """ Make a json file that contains the metadata of the simulation
#     """
#     model = inter.Model
#     snewpymodel = model.model
#     # where to save the json file
#     try:
#         store_at = model.config['wfsim']['sim_folder']
#     except Exception as e:
#         print(f"WFSim / sim_folder could not be found, storing the metadata in cwd,\n{e}")
#         store_at = "./"
#     # Check if json exists, create if not
#     output_json = os.path.join(store_at, jsonfilename)
#     # os.makedirs(output_json, exist_ok=True)
#
#     # create some metadata
#     meta = {'User': model.user, 'Storage': model.proc_loc, 'Model Name': model.model_name,
#             'Sim File': model.object_name,
#             'Time Range': f"{model.time_range[0]}, {model.time_range[1]}"}
#     # metadata from the snewpy model
#     for k, v in snewpymodel.metadata.items():
#         if isinstance(v, astropy.units.quantity.Quantity):
#             v = f"{v}"
#         meta[k] = v
#     meta['Model File'] = getattr(snewpymodel, "filename", "Unknown Snewpy Model Name")
#     meta['Duration'] = f"{np.round(np.ptp(snewpymodel.time), 2)}"
#     # metadata from the interaction object
#     meta['Interaction File'] = inter.interaction_name
#     meta['Nuclei Name'] = inter.Nuclei_name
#     meta['Isotope Name'] = inter.isotope_name
#     # metadata from the wfsim context
#     df = see_simulated_contexts(config_file=config_file, sim_id=sim_id)
#     df_dict = df.iloc[0].to_dict()
#     df_dict['context_name'] = df_dict['name']
#     df_dict['date_added'] = f"{df_dict['date_added']}"
#     df_dict.pop('sim_id')
#     df_dict.pop('name')
#     # make a json entry
#     json_entry = {sim_id: {"Model":meta, "Context": df_dict}}
#     # Append this simulation
#     if os.path.exists(output_json):
#         #read existing file and append new data
#         with open(output_json, "r") as f:
#             dictObj = json.load(f)
#         dictObj.update(json_entry)
#     else:
#         #create new json
#         dictObj = json_entry
#
#     #overwrite/create file
#     with open(output_json, "w") as f:
#         json.dump(dictObj, f, indent=4, sort_keys=True)

# def fetch_metadata(config_file, jsonfilename="simulation_metadata.json", full=False):
#     """ Fetch the metadata of the simulations
#     """
#     config = get_config(config_file)
#     store_at = config['wfsim']['sim_folder']
#     meta_file = os.path.join(store_at, jsonfilename)
#
#     try:
#         with open(meta_file, "r") as f:
#             dictObj = json.load(f)
#     except ValueError:
#         # Read the file
#         # Sometime there is a double closing bracket, fix it
#         with open(meta_file, 'r') as file:
#             lines = file.readlines()
#         # Check the last line
#         last_line = lines[-1].rstrip()
#         if last_line.endswith('}}'):
#             modified_last_line = last_line[:-1]
#             # Update the last line in the list of lines
#             lines[-1] = modified_last_line
#         # Write the modified content back to the file
#         with open(meta_file, 'w') as file:
#             file.writelines(lines)
#
#         with open(meta_file, "r") as f:
#             dictObj = json.load(f)
#
#     if full:
#         dd ={k: {**v['Context'], **v['Model']} for k, v in dictObj.items()}
#     else:
#         dd = {k: v['Model'] for k, v in dictObj.items()}
#     metaframe = pd.DataFrame(dd).T
#     return metaframe


def fetch_metadataframe(
    config_file, filename="simulation_metadata.csv", drop_duplicates=True
):
    """Fetch the metadata of the simulations"""
    config = get_config(config_file)
    store_at = config["wfsim"]["sim_folder"]
    meta_file = os.path.join(store_at, filename)
    metaframe = pd.read_csv(meta_file)
    collist = metaframe.columns.to_list()
    collist.remove("date simulated")
    if drop_duplicates:
        metaframe.drop_duplicates(subset=collist, keep="first", inplace=True)
    return metaframe


def split_sim_into_pieces(dataframe, timegap_seconds=100):
    """ split a multi-simulation in single file, into several dataframes
        if the user simulated N supernovae in one run, we can return a dictionary
        with N dataframes inside.
    """
    time_diffs = np.diff(dataframe['time'])

    # Define a threshold for the gap between clusters (assuming 2 minutes)
    gap_threshold = timegap_seconds * 1e9

    # Find the indices where the gap between timestamps exceeds the threshold
    cluster_indices = np.where(time_diffs > gap_threshold)[0] + 1

    # Split the DataFrame into separate clusters based on the identified indices
    clusters = np.split(dataframe, cluster_indices)

    # Optionally, you can store these clusters in a dictionary for easy access
    cluster_dict = {f'sim_{i}': cluster for i, cluster in enumerate(clusters)}
    return cluster_dict
