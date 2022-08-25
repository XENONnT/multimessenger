import strax
from numpy.lib.recfunctions import append_fields

export, __all__ = strax.exporter()
import numpy as np
import pandas as pd


@export
@strax.takes_config(
    strax.Option('sn_time_bin', default=7, type=int,
                 help='Time range which you want to find the rate over [s]'),
    strax.Option('min_area_electrons', default=57.8/32, type=int,
                 help='Minumum number of electrons in primary S2 [electrons]'),
    strax.Option('max_area_electrons', default=350/32, type=int,
                 help='Maximum number of electrons in primary S2 [electrons]'),
    strax.Option('min_width_ns', default=300, type=int,
                 help='Minumum width of S2 [ns]'),
    strax.Option('min_primary_size_pe', default=500, type=int,
                 help='Minumum size of primary S2 [PE]'),
    strax.Option('dt_cut_ns', default=2.5e6, type=np.float64,
                 help='Delay time cut value after primary S2'),
    strax.Option('dr_cut_cm', default=8.0, type=int,
                 help='Radius around primary S2 to exclude/cut away [cm]'),
    strax.Option('fv_cut_cm', default=60.2, type=int,
                 help='Radius of fiducial volume exclude/cut away [cm]'),

)


class SNPugins(strax.Plugin):
    """
    This is a test plugin
    """
    __version__ = '0.1.9'
    
    depends_on = ('peak_basics','peak_positions')
    provides = ('sn_rate')
    dtype = [
            ('sn_rate', np.float64,'Rate of peaks for SN [Hz]'),
            ('sn_rate_err', np.float64,'Error on rate of peaks for SN [sqrt(Hz)]'),
            ('time', np.int64,'Start time of bin since unix epoch [ns]'),
            ('endtime', np.int64,'End time of bin since unix epoch [ns]')
            ]
        
    def compute(self, peaks):   
        
        peaks_dt_dr_cuts = primary_matching(peaks, 
                                            self.config['min_primary_size_pe'], 
                                            self.config['dr_cut_cm'],
                                            self.config['dt_cut_ns']
                                           )
        
        
        rate, err, t_start, t_end = calculate_rate(peaks_dt_dr_cuts, 
                                                     self.config['sn_time_bin'], 
                                                     self.config['min_area_electrons'],
                                                     self.config['max_area_electrons'], 
                                                     self.config['min_width_ns'],
                                                     self.config['fv_cut_cm']
                                                    )
        
        df = pd.DataFrame()
        df['time'] = t_start
        df['endtime'] = t_end
        df['sn_rate'] = rate
        df['sn_rate_err'] = err
        
        return df


def calculate_rate(peaks, time_bin_s, min_area_electrons, max_area_electrons, min_width_ns, fv_cut_cm):
    se_gain = 32
    min_area_pe = se_gain*min_area_electrons
    max_area_pe = se_gain*max_area_electrons
    
    r_hot_cm = 10
    x_hot_cm = 7
    y_hot_cm = -15
    
    peaks_cut = peaks[ (peaks['type']==2) 
                      & (peaks['area'] > min_area_pe) 
                      & (peaks['area'] < max_area_pe)
                      & (peaks['range_50p_area'] > min_width_ns)
                      & ((peaks['x']**2 + peaks['y']**2) < fv_cut_cm**2)
                      & (((peaks['x'] - x_hot_cm)**2 + (peaks['y'] - y_hot_cm)**2) > r_hot_cm**2)
                     ]
    
    if len(peaks_cut) == 0:
        #'Do nothing'
        pass
        

    else:
        t_ns = peaks_cut['time'][0]

        time_s_to_ns = 1e9
        time_bin_ns = time_bin_s*time_s_to_ns

        rates = []
        rates_err = []
        t_start = []
        t_end = []

        while t_ns+time_bin_ns < peaks['time'][-1]:

            df_here = peaks_cut[(peaks_cut['time'] >= t_ns) & (peaks_cut['time'] < t_ns+time_bin_ns)]

            s2s = len(df_here)
            s2s_err = np.sqrt(s2s)
            rate = s2s/time_bin_s
            rate_err = s2s_err/time_bin_s

            rates.append(rate)
            rates_err.append(rate_err)
            t_start.append(t_ns)
            t_end.append(t_ns+time_bin_ns)

            if t_ns + time_bin_ns >= peaks['time'][-1]:
                t_ns = t_ns
            else:
                t_ns += time_bin_ns

        ## last bin when we are less than 5s from the end
        df_here = peaks_cut[(peaks_cut['time'] >= t_ns) & (peaks_cut['time'] <= peaks['time'][-1])]
        dt = (peaks['time'][-1] - t_ns)/1e9


        s2s = len(df_here)
        s2s_err = np.sqrt(s2s)
        rate = s2s/dt
        rate_err = s2s_err/dt

        rates.append(rate)
        rates_err.append(rate_err)
        t_start.append(t_ns)
        t_end.append(peaks['time'][-1])
    
    return rates, rates_err, t_start, t_end


def primary_matching(df, min_primary_size, dr_cut_cm, dt_cut_ns):
    print (len(df))
    
    max_drift_ns = 2.5e6
        
    #primary definition
    primary = df[(df['area'] > min_primary_size) & (df['type'] == 2)]
    primary.sort(order='center_time')

    #small S2s
    S2s = df[ (df['type'] == 2) & (df['area'] > 10) & (df['area'] < min_primary_size)]
    S2s = S2s[S2s['center_time'] >= min(primary['center_time'])]
    S2s.sort(order='center_time')

    #pairing
    prim_i = np.digitize(S2s['center_time'],primary['center_time']) - [1]*len(S2s)
    area_primary = primary['area'][prim_i]
    dt_primary = S2s['center_time'] - primary['center_time'][prim_i] 
    x_diff = S2s['x_mlp'] - primary['x_mlp'][prim_i]
    y_diff = S2s['y_mlp'] - primary['y_mlp'][prim_i]
    
    r_diff = np.nan_to_num(np.sqrt(x_diff**2 + y_diff**2))
    S2s = append_fields(S2s, 'r_diff', r_diff)
    S2s = append_fields(S2s, 'dt_primary', dt_primary)
    
    #delay time cut
    S2s_dtcut = S2s[S2s['dt_primary'] > dt_cut_ns]
    
    #position correlation cut
    S2s_drcut = S2s_dtcut[S2s_dtcut['r_diff'] > dr_cut_cm]
    
    return S2s_drcut