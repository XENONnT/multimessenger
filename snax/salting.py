""" This is to inject supernova neutrino signal into background runs
"""


import numpy as np
import pandas as pd
from tqdm.notebook import tqdm


def sample_by_removal(run_sq, Nsample, mindist):
    """ Sample from a given list, remove +/- mindist each time
        Might take long for long arrays
        Ensures min distance but does not guarantee exact Nsample
    """
    results = np.zeros(Nsample)
    run_sq_list = list(run_sq)
    for i in range(Nsample):
        if not (run_sq_list):
            break
        choice = np.random.choice(run_sq_list)
        results[i] = choice
        # remove all the data that are closer than mindist
        run_sq_list = [x for x in run_sq_list if (x <= choice - mindist) or (x >= choice + mindist)]

    return np.sort(results[results != 0]).astype(int)


def try_sample(df_bg, Nsample, mindist, trial=10, verbose=False):
    """ `sample_by_removal` can result in different sizes
        this function tries to sample `trial`-many times
        and returns the largest sample
    """
    run_sq = _get_run_intervals(df_bg, mindist)
    arrs = []
    for i in range(trial):
        result = sample_by_removal(run_sq, Nsample, mindist)
        arrs.append(result)
        if len(result) == Nsample:
            print(f"Found at {i}-th trial") if verbose else None
            return result
    lengths = [len(x) for x in arrs]
    max_i = lengths.index(max(lengths))
    print(f"Could not sample {Nsample} with minimum separation {mindist} after {trial} trials,\n\
    returning max sample with {len(arrs[max_i])} points")
    return arrs[max_i]


def _get_run_intervals(df_bg, mindist_sec):
    """ For an arbitrarily combined background runs
        Find the time intervals, and combine a continues `seconds` array
        Returns
        sampled times in seconds
    """
    run_ids = df_bg.run_id.unique()
    nruns = len(run_ids)
    # get the min time in sec (avoid first mindist seconds)
    intervals = np.zeros((nruns, 4), dtype=int)

    for i, r in enumerate(run_ids):
        start_sec = np.min(df_bg[df_bg["run_id"] == r]['time']) * 1e-9 + mindist_sec
        end_sec = np.max(df_bg[df_bg["run_id"] == r]['endtime']) * 1e-9 - mindist_sec
        livetime = end_sec - start_sec
        intervals[i] = [start_sec, end_sec, r, livetime]

    # make a running seconds for each interval and concatanate
    # imagine runs with seconds from [1,15], [20,35], [45,60]
    # below returns an array with [1,2,3,4..15,20,21,23..35,45,46,47..60]
    _run_sequence = []
    for i in range(nruns):
        _run_sequence.append(np.arange(intervals[i, 0], intervals[i, 1] + 1))
    run_sequence = np.concatenate(_run_sequence).flatten()
    return run_sequence


def make_time_index(dataframe, sort_by='time', inplace=False):
    """ For rolling time windows, turn the data into time-indexed frame
        Also sort the times to fix the them after injection
    """
    if inplace:
        _dataframe = dataframe
    else:
        _dataframe = dataframe.copy()

    _dataframe.sort_values(by=sort_by, inplace=True)
    _dataframe.reset_index(drop=True, inplace=True)
    _dataframe["time_index"] = pd.to_datetime(_dataframe[sort_by], unit='ns', origin='unix')
    _dataframe.set_index("time_index", inplace=True)
    return _dataframe


def select_runids(runids, Nsim=10, verbose=True):
    """ Select Nsim many simulations from all available simulated data,
        use select_str and exclude_str to select one type of model
    """
    runids = np.unique(runids)
    selected_sims = np.random.choice(runids, Nsim)
    if verbose:
        print(f"{Nsim} simulation ids selected:\n{selected_sims}")
    return selected_sims


def inject_in_SR0(bg_data, sim_data, Npoints,
                  return_selected_sims=False, min_seperation_sec=240):
    """ Inject `Npoints` in `bg_data`, selecting the injected data from `sim_data`
        without or `with_replacement` (has to be True if you request more than 21 points)
        The injection points are selected `min_seperation_sec` seconds away from the edges and
        from each other.
        `return_selected_sims` for further comparison
    """
    print(f"\t Sampling {Npoints} time-points..")
    # sample times inject SN signal
    sample = try_sample(bg_data, Npoints, mindist=min_seperation_sec, trial=20)
    sampled_injection_points = sample * int(1e9)  # needs to be in nanoseconds

    # might get less
    Npoints = len(sampled_injection_points)
    sim_ids_selected = select_runids(sim_data.run_id, Nsim=Npoints, verbose=False)

    print(f"\t Selecting simulations..")
    subsims = sim_data[sim_data['run_id'].isin(sim_ids_selected)]
    assert len(sampled_injection_points) == len(sim_ids_selected), "mismatching injection points and simulations"

    print(f"\t Adjusting time columns for injection..")
    # change the columns with the "time" for injection
    # there can be more than one selection for the same data, so it needs to be added each time
    _subsims_2 = []
    for i, runid in tqdm(enumerate(sim_ids_selected), total=len(sim_ids_selected)):
        dd = subsims[subsims['run_id'] == runid].copy()
        mintime_subsim = np.min(dd['time'])
        dd.loc[:, 'time'] = dd.loc[:, 'time'] - mintime_subsim  # from 0 to t, in ns
        dd.loc[:, 'time'] = dd.loc[:, 'time'] + sampled_injection_points[i]

        dd.loc[:, 'endtime'] = dd.loc[:, 'endtime'] - mintime_subsim  # from 0 to t, in ns
        dd.loc[:, 'endtime'] = dd.loc[:, 'endtime'] + sampled_injection_points[i]

        dd.loc[:, 'time'] = dd.loc[:, 'center_time'] - mintime_subsim  # from 0 to t, in ns
        dd.loc[:, 'time'] = dd.loc[:, 'center_time'] + sampled_injection_points[i]

        _subsims_2.append(dd)

    subsims_2 = pd.concat(_subsims_2)
    subsims_2 = subsims_2.astype({"time": int, "endtime": int, "run_id": str})

    # combine and make a new frame
    print(f"\t Creating an injected & combined data..")
    combined_df = pd.concat([bg_data, subsims_2])
    combined_df = combined_df.sort_values(by='time').reset_index(drop=True)
    make_time_index(combined_df, inplace=True)
    combined_df = combined_df.astype({"time": int, "endtime": int, "run_id": str})

    if return_selected_sims:
        # return subsims
        make_time_index(subsims_2, inplace=True)
        return combined_df, (sim_ids_selected, subsims_2), sampled_injection_points
    return combined_df, sampled_injection_points





