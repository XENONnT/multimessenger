"""
Author: Melih Kara kara@kit.edu

The auxiliary PLOTTING tools that are used within the SN signal generation, waveform simulation

"""

import matplotlib.pyplot as plt
import numpy as np
import straxen
from matplotlib.colors import LogNorm
import multihist as mh


def quality_plot(event_info):
    """ Quick diagnostics plot for event level data
    """
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.6, hspace=0.45)
    axes[0, 0].set_xscale('log')
    axes[0, 0].hist(event_info['s1_area'], bins=np.logspace(-0.2, 2, 100), histtype='step', color='magenta', lw=3)
    axes[0, 0].tick_params(axis='x', labelcolor='teal', labelsize=12)
    axes[0, 0].set_xlabel('S1 Area [P.E.]', color='teal', fontsize=12)

    axes00twin = axes[0, 0].twiny()
    axes00twin.set_xscale('log')
    axes00twin.hist(event_info['s2_area'], bins=np.logspace(1, 4, 100), histtype='step', color='teal', lw=3)
    axes00twin.tick_params(axis='x', labelcolor='magenta', labelsize=12)
    axes00twin.set_xlabel('S2 Area [P.E.]', color='magenta', fontsize=12)

    axes[0, 1].set_xscale('log')
    axes[0, 1].hist(event_info['s1_range_50p_area'], bins=np.logspace(1, 4, 100), histtype='step', color='magenta',
                    lw=3)
    axes[0, 1].tick_params(axis='x', labelcolor='magenta', labelsize=12)
    axes[0, 1].set_xlabel('S1 50% width [ns]', color='magenta', fontsize=12)

    axes01twin = axes[0, 1].twiny()
    axes01twin.set_xscale('log')
    axes01twin.hist(event_info['s2_range_50p_area'], bins=np.logspace(2, 5, 100), histtype='step', color='teal', lw=3)
    axes01twin.tick_params(axis='x', labelcolor='teal', labelsize=12)
    axes01twin.set_xlabel('S2 50% width [ns]', color='teal', fontsize=12)

    axes[1, 0].set_title('area vs 50p width (S1)', color='teal', fontsize=13)
    axes[1, 0].hist2d(event_info['s1_area'], event_info['s1_range_50p_area'],
                      bins=(np.logspace(-0.2, 2, 100), np.logspace(1, 4, 100)), cmap='Blues', norm=LogNorm())
    axes[1, 0].tick_params(axis='both', labelcolor='teal', labelsize=12)
    axes[1, 0].set_xlabel('S1 area [P.E.]', color='teal', fontsize=12)
    axes[1, 0].set_ylabel('S1 50% width [ns]', color='teal', fontsize=12)
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')

    axes[1, 1].set_title('area vs 50p width (S2)', color='magenta', fontsize=13)
    axes[1, 1].hist2d(event_info['s2_area'], event_info['s2_range_50p_area'],
                      bins=(np.logspace(1, 4, 100), np.logspace(1, 5, 100)), cmap='Reds', norm=LogNorm())
    axes[1, 1].tick_params(axis='both', labelcolor='magenta', labelsize=12)
    axes[1, 1].set_xlabel('S2 area [P.E.]', color='magenta', fontsize=12)
    axes[1, 1].set_ylabel('S2 50% width [ns]', color='magenta', fontsize=12)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')

    axes[0, 2].hist2d(event_info['x'], event_info['y'], bins=(np.linspace(-70, 70, 50), np.linspace(-70, 70, 50)),
                      norm=LogNorm())
    axes[0, 2].set_aspect('equal')
    axes[1, 2].hist2d(event_info['r'], event_info['z'],
                      bins=(np.linspace(-5, 80, 100), np.linspace(-160, 10, 100)), norm=LogNorm())
    ##
    axes[2, 0].hist(event_info['s1_area_fraction_top'], bins=np.linspace(-0.1, 1.1, 100), histtype='step',
                    color='magenta', lw=3, density=True)
    axes[2, 0].tick_params(axis='x', labelcolor='teal', labelsize=12)
    axes[2, 0].set_xlabel('S1 Area Fraction Top', color='teal', fontsize=12)

    axes20twin = axes[2, 0].twiny()
    axes20twin.hist(event_info['s2_area_fraction_top'], bins=np.linspace(-0.1, 1.1, 100), histtype='step', color='teal',
                    lw=3, density=True)
    axes20twin.tick_params(axis='x', labelcolor='magenta', labelsize=12)
    axes20twin.set_xlabel('S2  Fraction Top', color='magenta', fontsize=12)
    ##
    axes[2, 1].hist2d(event_info['s1_area'], event_info['s2_area'],
                      bins=(np.logspace(-0.2, 2, 100), np.logspace(1.8, 3.7, 100)), norm=LogNorm())
    axes[2, 1].tick_params(axis='x', labelcolor='teal', labelsize=12)
    axes[2, 1].tick_params(axis='y', labelcolor='magenta', labelsize=12)
    axes[2, 1].set_xlabel('S1 area [P.E.]', color='teal', fontsize=12)
    axes[2, 1].set_ylabel('S2 area [P.E.]', color='magenta', fontsize=12)
    axes[2, 1].set_xscale('log')
    axes[2, 1].set_yscale('log')

    ##
    axes[2, 2].set_title('Counts above S2 threshold', fontsize=12)
    counts_above_i = []
    s2range = np.linspace(event_info['s2_area'].min(), 1500, 50)
    for i, thr in enumerate(s2range):
        counts_above_i.append(len(event_info['s2_area'][event_info['s2_area'] > thr]))
    axes[2, 2].loglog(s2range, counts_above_i, lw=3)
    axes[2, 2].set_xlabel('S2 area [P.E.]')
    axes[2, 2].set_ylabel('counts')


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

