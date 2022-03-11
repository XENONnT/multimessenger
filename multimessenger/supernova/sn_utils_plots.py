"""
Author: Melih Kara kara@kit.edu

The auxiliary PLOTTING tools that are used within the SN signal generation, waveform simulation

"""

import matplotlib.pyplot as plt
import numpy as np
import straxen


def quality_plot(ev):
    """ Quick diagnostics plot

    """
    plt.figure(figsize=(15,10))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.subplot(321); plt.title('s1 area')
    plt.hist(ev['s1_area'], bins = 200, histtype = 'step')
    
    plt.subplot(322); plt.title('s2 area')
    plt.hist(ev['s2_area'], bins = 200, histtype = 'step')
    
    plt.subplot(323); plt.title('s2 width')
    plt.hist(ev['s2_range_50p_area'], bins = 200, histtype = 'step')
    
    plt.subplot(324); plt.title('counts above area threshold')
    counts_above_i = []
    s2range = np.linspace(ev['s2_area'].min(), 1500, 50)
    for i, thr in enumerate(s2range):
        counts_above_i.append(len(ev['s2_area'][ev['s2_area']>thr]))
    plt.semilogy(s2range, counts_above_i)
    plt.xlabel('S2 area')

    plt.subplot(325); plt.title('z [cm]')
    plt.hist(ev['z'][ev['z']<0], bins = 200, histtype = 'step')
    plt.axvline(-straxen.tpc_z, ls = '--', color = 'k')
    
    plt.subplot(326)
    plt.gca().set_aspect('equal')
    s1 = ev['s1_area']
    s2 = ev['s2_area']
    mask = (s1>0) & (s2>0) 
    kwargs = dict(norm=LogNorm(), bins=100) # range=[[0, 3_000], [0, 20_000]],
    mh_cut = mh.Histdd(s1[mask], s2[mask]/100, **kwargs)
    mh_cut.plot(log_scale=True, cblabel='count',
            cmap=plt.get_cmap('jet'), alpha=0.7,  
            colorbar_kwargs=dict(orientation="vertical", 
                                 pad=0.05,
                                 aspect=30, 
                                 fraction=0.1))
    plt.xlabel('s1 area [P.E.]'); plt.ylabel('s2 area [100 P.E.]')


def plot_xy_rz(evt):
    """ Plot events in xy and R-z planes

    """
    plt.figure(figsize=(12,4))
    plt.subplots_adjust(wspace=0.9)
    plt.subplot(121)
    plt.hist2d(evt['x'],evt['y'], range = ((-70,70),(-70,70)),cmin = 1,bins = 100) # , norm = LogNorm()
    plt.xlabel('x [cm]')
    plt.ylabel('y [cm]')
    plt.gca().set_aspect('equal')
    plt.colorbar()
    plt.subplot(122)
    plt.hist2d(np.power(evt['r'],2),evt['z'], range = ((0,70**2),(-160,10)),cmin = 1,bins = 100) # , norm = LogNorm()
    plt.colorbar()
    plt.xlabel('r$^2$ [cm]')
    plt.ylabel('z [cm]')
    plt.show()


def compare_features(peak_basics, peaks_run):
    """ compare peak area, width and aft's

    """
    s2_sn = peak_basics[(peak_basics['type']==2)&(peak_basics['area']<2000)]
    s1_sn = peak_basics[peak_basics['type']==1]

    s2_runs = peaks_run[(peaks_run['type']==2)&(peaks_run['area']<2000)]
    run_area = s2_runs['area']
    run_width = s2_runs['range_50p_area']
    run_aft = s2_runs['area_fraction_top']
    sn_area = s2_sn['area']
    sn_width = s2_sn['range_50p_area']
    sn_aft = s2_sn['area_fraction_top']

    fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(17,13))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    ax[0,0].hist2d(np.log10(run_area[(run_area>0)&(run_width>0)]), 
                   np.log10(run_width[(run_area>0)&(run_width>0)]), 
                   bins=(200,200), norm=LogNorm(), cmap='Reds')
    ax[0,0].hist2d(np.log10(sn_area), np.log10(sn_width), bins=(200,200), norm=LogNorm(), alpha=0.7)
    ax[0,0].set_xscale('log') ; ax[0,0].set_yscale('log')
    ax[0,0].set_xlabel('log(S2 area)'); ax[0,0].set_ylabel('log(S2 width)')

    ax[0,1].hist2d(run_area, run_aft, bins=(200,200), norm=LogNorm(), cmap='Reds')
    ax[0,1].hist2d(sn_area, sn_aft, bins=(200,200), norm=LogNorm(), alpha=0.6)
    ax[0,1].set_xlabel('S2 Area [P.E.]'); ax[0,1].set_ylabel('S2 AFT')
    ax[0,1].axhline(0.68)

    ax[0,2].hist2d(run_width, run_aft, bins=(200,200), norm=LogNorm(), cmap='Reds')
    ax[0,2].hist2d(sn_width, sn_aft, bins=(200,200), norm=LogNorm(), alpha=0.6)
    ax[0,2].set_xlabel('width')
    ax[0,2].set_ylabel('AFT')

    ax[1,0].hist(run_area, bins=50, histtype='step', density=True, label='BG')
    ax[1,0].hist(sn_area, bins=50, histtype='step', density=True, label='SN')
    ax[1,0].set_xlabel('S2 area')
    ax[1,0].set_yscale('log')

    ax[1,1].hist(run_aft, bins=50, histtype='step', density=True, label='BG')
    ax[1,1].hist(sn_aft, bins=50, histtype='step', density=True, label='SN')
    ax[1,1].set_xlabel('AFT')
    ax[1,1].set_yscale('log')

    ax[1,2].hist(run_width, bins=50, histtype='step', density=True, label='BG', range=(0,15_000))
    ax[1,2].hist(sn_width, bins=50, histtype='step', density=True, label='SN', range=(0,15_000))
    ax[1,2].set_xlabel('width')
    ax[1,2].set_yscale('log')

    for i in range(3):
        i = int(i)
        ax[1,i].legend()

