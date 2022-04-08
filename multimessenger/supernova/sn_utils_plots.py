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


def compare_peaks(peaks_sim, peaks_data,
                  area=(60,2000),
                  width50p=(120, 30000),
                  aft=(0.35, 0.9)):
    """ Compare Peaks for Simulation and Data
        *** S2 Only
        :param area, width50p, aft : `tuple` ranges to draw lines
    """
    s2_sim = peaks_sim[peaks_sim['type'] == 2]
    s2_data = peaks_data[peaks_data['type'] == 2]

    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(18, 10))
    plt.subplots_adjust(wspace=0.5, hspace=0.45)

    kwargs = dict(bins=np.logspace(1, 4, 100), cumulative=True, histtype='step', lw=3, density=True)
    axes[0, 0].hist(s2_sim['area'], label='sim', color='red', **kwargs)
    axes[0, 0].hist(s2_data['area'], label='data', color='blue', **kwargs)
    axes[0, 0].tick_params(axis='both', labelsize=12)
    axes[0, 0].set_xlabel(f'S2 Area [P.E.]', fontsize=12)
    axes[0, 0].set_xscale('log')
    axes[0, 0].legend(loc='lower right', fontsize=12)

    #
    kwargs = dict(bins=np.logspace(1.5, 5, 100), cumulative=True, histtype='step', lw=3, density=True)
    for ax, p, a in zip([axes[0, 1], axes[0, 1]], ['50', '90'], [0.5, 1]):
        ax.hist(s2_sim[f'range_{p}p_area'], label=f'sim {p}%', color='red', alpha=a, **kwargs)
        ax.hist(s2_data[f'range_{p}p_area'], label=f'data {p}%', color='blue', alpha=a, **kwargs)
        ax.tick_params(axis='both', labelsize=12)
        ax.set_xlabel(f'S2 width [ns]', fontsize=12)
        ax.set_xscale('log')
        ax.legend(loc='upper left', fontsize=12)

    #
    kwargs = dict(bins=np.linspace(0, 1.1, 100), cumulative=True, histtype='step', lw=3, density=True)
    axes[0, 2].hist(s2_sim['area_fraction_top'], label='sim', color='red', **kwargs)
    axes[0, 2].hist(s2_data['area_fraction_top'], label='data', color='blue', **kwargs)
    axes[0, 2].tick_params(axis='both', labelsize=12)
    axes[0, 2].set_xlabel(f'S2 area_fraction_top', fontsize=12)
    axes[0, 2].legend(loc='upper left', fontsize=12)
    #
    axes[1, 0].hist2d(s2_data['area'], s2_data['range_50p_area'], bins=(np.logspace(0, 5, 100),
                                                                        np.logspace(1.5, 5, 100)), norm=LogNorm())
    axes[1, 0].hist2d(s2_sim['area'], s2_sim['range_50p_area'], bins=(np.logspace(0, 5, 100),
                                                                      np.logspace(1.5, 5, 100)), norm=LogNorm(),
                      alpha=0.8, cmap='Reds_r')
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('S2 area [P.E.]')
    axes[1, 0].set_ylabel('S2 width')
    #
    axes[1, 1].hist2d(s2_data['range_50p_area'], s2_data['area_fraction_top'],
                      bins=(np.logspace(1.5, 5, 100), np.linspace(-0.1, 1.1, 100)), norm=LogNorm())
    axes[1, 1].hist2d(s2_sim['range_50p_area'], s2_sim['area_fraction_top'],
                      bins=(np.logspace(1.5, 5, 100), np.linspace(-0.1, 1.1, 100)), norm=LogNorm(), alpha=0.8,
                      cmap='Reds_r')
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_xlabel('S2 width [ns]')
    axes[1, 1].set_ylabel('S2 aft')
    #
    axes[1, 2].hist2d(s2_data['area'], s2_data['area_fraction_top'],
                      bins=(np.logspace(0, 5, 100), np.linspace(-0.1, 1.1, 100)), norm=LogNorm())
    axes[1, 2].hist2d(s2_sim['area'], s2_sim['area_fraction_top'],
                      bins=(np.logspace(0, 5, 100), np.linspace(-0.1, 1.1, 100)), norm=LogNorm(), alpha=0.8,
                      cmap='Reds_r')
    axes[1, 2].set_xscale('log')
    axes[1, 2].set_xlabel('S2 area [P.E.]')
    axes[1, 2].set_ylabel('S2 aft')

    # random cuts
    # area cuts
    axes[0, 0].axvline(area[0], color='k', lw=3, ls='--')
    axes[0, 0].axvline(area[1], color='k', lw=3, ls='--')
    axes[1, 0].axvline(area[0], color='red', lw=3)
    axes[1, 0].axvline(area[1], color='red', lw=3)
    axes[1, 2].axvline(area[0], color='red', lw=3)
    axes[1, 2].axvline(area[1], color='red', lw=3)

    # width cuts
    axes[0, 1].axhline(width50p[0], color='red', lw=3, ls='--')
    axes[0, 1].axhline(width50p[1], color='red', lw=3, ls='--')
    axes[1, 0].axhline(width50p[0], color='red', lw=3)
    axes[1, 0].axhline(width50p[1], color='red', lw=3)
    axes[1, 1].axvline(width50p[0], color='red', lw=3)
    axes[1, 1].axvline(width50p[1], color='red', lw=3)

    # aft cuts
    axes[0, 2].axvline(aft[0], color='k', lw=3, ls='--')
    axes[0, 2].axvline(aft[1], color='k', lw=3, ls='--')
    axes[1, 1].axhline(aft[0], color='red', lw=3)
    axes[1, 1].axhline(aft[1], color='red', lw=3)
    axes[1, 2].axhline(aft[0], color='red', lw=3)
    axes[1, 2].axhline(aft[1], color='red', lw=3)







