"""
Author: Melih Kara kara@kit.edu

The auxiliary PLOTTING tools that are used within the SN signal generation, waveform simulation

"""
from .libraries import *

def quality_plot(ev):
    plt.figure(figsize=(15,10))
    plt.subplots_adjust(wspace=0.2, hspace=0.5)
    plt.subplot(321); plt.title('s1 area')
    plt.hist(ev['s1_area'], bins = 200, histtype = 'step')
    
    plt.subplot(322); plt.title('s2 area')
    plt.hist(ev['s2_area'], bins = 200, histtype = 'step')
    
    plt.subplot(323); plt.title('s2 width')
    plt.hist(ev['s2_range_50p_area'], bins = 200, histtype = 'step')
    
    plt.subplot(324); plt.title('counts above area threshold') #plt.title('dt [us]')
#     plt.hist(ev['drift_time']/1000, bins = 200, histtype = 'step')
    counts_above_i = []
    s2range = np.linspace(ev['s2_area'].min(), 1500, 50)
    for i, thr in enumerate(s2range):
        counts_above_i.append(len(ev['s2_area'][ev['s2_area']>thr]))
    plt.semilogy(s2range, counts_above_i)
    plt.xlabel('S2 area')

    plt.subplot(325); plt.title('z [cm]')
    plt.hist(ev['z'][ev['z']<0], bins = 200, histtype = 'step')
    plt.axvline(-straxen.tpc_z, ls = '--', color = 'k')
    
    plt.subplot(326);
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
                                 fraction=0.1));
    plt.xlabel('s1 area [P.E.]'); plt.ylabel('s2 area [100 P.E.]');
    
def plot_xy_rz(evt):
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