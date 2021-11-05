#!/usr/bin/python

"""
Last Update: 01-09-2021
-----------------------
Supernova Plotter
Author: Melih Kara kara@kit.edu

Notes
-----
# pickling images https://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle/
# extracting data fig_handle.axes[0].lines[0].get_data()
"""
from aux_scripts.constants import *
from aux_scripts.set_params import *
import os
os.system('./aux_scripts/set_file_paths.py')


class Plotter:
    """ Plotter Class for SN data
    """
    def __init__(self,
                 Model, 
                 figsize=(17,4), 
                 cmap='jet'):
        """
        Parameters
        ----------
        Model : obj
            Supernova_Models object
        figsize : tuple, optional
            default figure size
        cmap : str, optional
            default colormap
        """
        self.Model = Model
        self.figsize = figsize
        self.cmap    = matplotlib.cm.get_cmap(cmap)
        self.colors  = ['r','g','b','orange','teal','brown', 'gold']
        self.__version__ = "0.0.2"

    def _get_figure(self, figsize=None, ftype='vertical'):
        """
        get a figure, axes instance
        ftype can be 'vertical', 'horizontal' or 'single'
        """
        figsize = figsize or self.figsize
        if ftype.lower()=='vertical':
            figsize = (min(figsize),max(figsize))
            ncol, nrow = 1,3
            sharex,sharey = True,False
        if ftype.lower()=='horizontal':
            figsize = (max(figsize),min(figsize))
            ncol, nrow = 3,1
            sharex,sharey = False,True
        if ftype.lower()=='single':
            figsize = (max(figsize),min(figsize))
            ncol, nrow = 1,1
            sharex,sharey = False,False
        fig, axes = plt.subplots(ncols=ncol,nrows=nrow, 
                                 figsize=figsize, sharex=sharex, sharey=sharey)
        return fig, axes

    def plot_3cols(self, _y, _x, 
                   figsize=None, 
                   cmap=None, 
                   ftype='vertical', 
                   average=False, 
                   ave_over='time'):
        """ Generic 3 axes plotter
            Can plot raw or integrated spectra, by checking the ndim
            3 plots can be vertically or horizontally aligned
            In the raw spectra (i.e. from different energies), 
            the plot can be averaged over these energies
        """
        fig, axes = self._get_figure(figsize,ftype)
    
        test_key = list(_y.keys())[0]
        ydim = _y[test_key].ndim
        if ydim == 2:    # i.e. data has multiple bins
            return self._plot_raw_data(fig, axes,_y, _x, self.cmap, ave_over, average)
        elif ydim == 1:  # i.e. data is integrated over one axis
            return self._plot_integrated(fig, axes, _y, _x, self.cmap)

    def _plot_raw_data(self, fig, axes ,_y, _x, cmap, ave_over, average):
        """ plot the raw spectra
            One line for each energy/time bin
        """
        if average==True:
            axis_to_average = 1 if ave_over=='time' else 0
            if type(axes)==np.ndarray:  zipped = zip(axes,_y.items())
            if type(axes)!=np.ndarray:  zipped = zip([axes]*3,_y.items())

            for i, (ax, (name, data)) in enumerate(zipped):
                data = np.average(data, axis=axis_to_average)
                ax.loglog(_x , data, ls ='-', c=self.colors[i], label=name)
            return fig, ax

        for ax, (name, data) in zip(axes, _y.items()):
            if ave_over.lower()=='time':
                frac = len(self.Model.mean_E)
                for i, _ in enumerate(self.Model.mean_E):
                    if i%2==1: continue # skip every other line to reduce crowd
                    left_bound = int(self.Model.E_bins_left[i])
                    right_bound = int(self.Model.E_bins_right[i])
                    ax.loglog(_x , data[:,i], ls ='-',c = cmap(i/frac), 
                        label = f'[{left_bound}:{right_bound}] MeV')
            else:
                frac = len(self.Model.t)
                for i, _ in enumerate(self.Model.t):
                    if i%30==1: continue
                    ax.loglog(_x , data[i,:], ls ='-',c = cmap(i/frac))
            ax.set_title(name, fontdict=font_medium)
        return fig, axes

    def _plot_integrated(self, fig, axes ,_y, _x, cmap):
        """ Plot time integrated spectra 
        """
        if type(axes) == np.ndarray:
            for i, (ax, (name, data)) in enumerate(zip(axes, _y.items())):
                ax.loglog(_x , data, ls ='-', c=self.colors[i], label=name, lw=3)
                ax.legend(fontsize=16)
        else:
            for i, (name, data) in enumerate(_y.items()):
                axes.loglog(_x , data, ls ='-', c=self.colors[i], label=name, lw=3)
                axes.legend(fontsize=16)
        return fig,axes

    def adjust_xylabels(self, ftype, axs, xlabel, ylabel, xscale, yscale):
        if ftype=='single':
            axs.set_xlabel(xlabel, fontdict=font_medium)
            axs.set_ylabel(ylabel, fontdict=font_medium)
            axs.legend(ncol=2, fontsize=15);
            axs.set_xscale(xscale); axs.set_yscale(yscale);
        else:
            if ftype=='vertical':
                axs[0].set_xscale(xscale); axs[0].set_yscale(yscale);
                axs[1].set_ylabel(ylabel, fontdict=font_medium)
                axs[2].set_xlabel(xlabel, fontdict=font_medium)
            else:
                axs[0].set_xscale(xscale); axs[0].set_yscale(yscale);
                axs[0].set_ylabel(ylabel, fontdict=font_medium)
                for ax in axs:
                    ax.set_xlabel(xlabel, fontdict=font_medium)
            axs[2].legend(ncol=2, fontsize=15);
        return axs

    def plot_data(self, x='time', y='Luminosity', ftype='horizontal', 
                   xscale='log',yscale='log',**kwargs):
        """ Plot Luminosities/Number Fluxes vs Time/Energies
        Arguments:
        ---------
        x        : str
            'time' or 'energy', what quantity to put on x axis
        y        : str
            'luminosity' or 'number', what quantity to put on y axis
        vertical : bool
            Only relevant if single=False
            The layout of the plot, vertical or horizontal
        single   : bool
            To plot all the flavors in a single frame
        average  : bool
            whether or not to average a given data over the other
            i.e. when x=energy, and average is True, it averages over time
            and when x=time, and average is True, it averages over energies
        **kwargs
        """
        if x.lower()=='time': _x, xlabel = self.Model.t , 'time [s]'
        elif x.lower()=='energy':  _x, xlabel = self.Model.mean_E , r'$\langle E_\nu \rangle$ [MeV]'
        else:  return print(f'{x} not recognized')

        if y.lower()=='luminosity': _y, ylabel = self.Model.L_list, 'Luminosity [erg/s/MeV]'
        elif y.lower()=='number': _y, ylabel = self.Model.nu_list, r'$\nu$ number [cts/s/MeV]'
        else: return print(f'{y} not recognized')

        # only if single frame wanted
        average = True if ftype.lower()=='single' else False

        fig, axs = self.plot_3cols(_y, _x,  ftype=ftype, average=average, ave_over=x, **kwargs)
        axs = self.adjust_xylabels(ftype, axs, xlabel, ylabel, xscale, yscale)
        return fig, axs

    def plot_data_integrated(self, t0=0, tf=10, 
                             at_detector=False,
                             ftype='horizontal', 
                             total=False,
                             xscale='log',
                             yscale='log', **kwargs):
        ''' plots the Flux for different energy bins
            integrated over given time. All parameters are optional.
            Parameters
            ----------
            t0,tf : float
                start,end time for the neutrino signal in seconds.
            at_detector : bool
                whether to plot data at the detector or raw.
            ftype : str
                'horizontal', 'vertical', or 'single'
            total : bool
                if True multiply y axis with mean energies
                so its the total counts
            xscale, yscale : str
                'log' or 'linear' axis scales.
        '''
        xlabel = r'E$_\nu$ [MeV]'
        ylabel = r'$\nu$ number [cts/MeV]'
        number_flux = self.Model.get_integ_fluxes(at_detector,t0,tf)
        # multiply y axis with mean energies if total
        if total: 
            number_flux = {nu : number_flux[nu]*self.Model.mean_E for nu in number_flux.keys()}
            ylabel = r'Total $\nu$ number [cts]'
        if at_detector: ylabel = 'Total $\\nu$ number at the detector\n[cts/cm$^2$]'
        fig, axs = self.plot_3cols(number_flux, self.Model.mean_E, ftype=ftype, **kwargs)
        plt.suptitle(f'Neutrino Flux integrated over ({t0:.1f}-{tf:.1f}) s', fontdict=font_medium)
        axs = self.adjust_xylabels(ftype, axs, xlabel, ylabel, xscale, yscale)
        return fig, axs

    def plot_recoil_spectra(self, spectra=None, figsize=None): 
        """
        TODO: Correct truncate
        TODO different E_r acceptance
        """
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        recoil_energies = self.Model.recoil_en

        if type(spectra) == type(None):  # if no spectra given check if it exists
            try: spectra = self.Model.rate1D
            except: 
                print(f'No rate found \n>Running get_recoil_spectra() with defaults') 
                spectra = self.get_recoil_spectra()
                
        for name, data in spectra.items():
            ax.plot(recoil_energies, data,  label=name, lw=3)
        ax.plot(self.Model.recoil_en, self.Model.total_rate1D, 'k', ls=':',label="Total", lw=3)
        ax.legend()
        ax.set_title(f'SN recoil spectrum (integrated between {self.Model.t0}s {self.Model.tf}s)')
        ax.set_xlabel("Neutrino-xenon recoil energy [keVnr]")
        ax.set_ylabel("Diff rate [events/tonne/keV]")
        return fig, ax

    def plot_rate_above_threshold(self, figsize=None):
        try: rate = self.Model.rate1D
        except: 
            print(f'No rate found \n>Running get_recoil_spectra() with defaults') 
            rate = self.get_recoil_spectra()     
        recoil_energies = self.Model.recoil_en
        E_thr =  np.array([trapz(self.Model.total_rate1D[i:], recoil_energies[i:]) for i,foo in enumerate(recoil_energies)])

        figsize = figsize or self.figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(recoil_energies, E_thr, c='r', lw=3)
        ax.set_xlabel("Neutrino-xenon recoil energy threshold [keVnr]")
        ax.set_ylabel("Rate above threshold \n[events/tonne/keV]")
        return fig, ax

    def plot_sampled_energies(self, 
                              x='energy', 
                              N_sample=1000000, 
                              xscale='linear', 
                              yscale='linear', 
                              bins=250, 
                              figsize=None): # TODO: Tidy up it a bit
        '''
        Sample from recoil energies and plot 

        Return
        ------
            tuple : fig, ax, sampled energies
        '''
        figsize = figsize or self.figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        if x.lower()=='energy': spectrum, xaxis = self.Model.total_rate1D, self.Model.recoil_en
        elif x.lower()=='time':
            # spectrum_Er['Total'] is the same as self.total_rate1D (if run for all time range)
            # it is a seperate if, in case one wants to plot without having 2D data
            spectrum_Er, spectrum_t = self.Model._get_1Drates_from2D()
            spectrum = spectrum_t['Total']
            xaxis = self.Model.t
        else: return print('choose x=time or x=energy')
            
        if xscale=='log': bins = np.logspace(np.log10(xaxis[xaxis>0].min()),np.log10(xaxis.max()), bins)
        else: bins = bins
        
#         Er_sampled = self.Model._inverse_transform_sampling(xaxis, spectrum, N_sample)
        Er_sampled = self.Model.sample_from_recoil_spectrum(x=x.lower(), N_sample=N_sample)
        E_thr_tot = np.trapz(spectrum, xaxis)
        pdf_diff_rate = spectrum/E_thr_tot
        ax.plot(xaxis, pdf_diff_rate, label='pdf', lw=3, alpha=0.7)
        ax.hist(Er_sampled, histtype='step', lw=3, bins=bins, alpha=1, density=True, label='Sampled Points');
        ax.legend(fontsize=16)

        xl = 'time [s]' if x.lower()=='time' else r'E$_r$'
        ax = self.adjust_xylabels('single', ax, xl, 'counts', xscale, yscale)
        return fig, ax, Er_sampled

    def _integrated_spectra2D(self, t0, tf, figsize):
        # TODO: Instead use Supernova_Models._get_1Drates_from2D
        # Take data
        tbins, Ebins = self.Model.t, self.Model.recoil_en  
        rates2D = self.Model.rates2D

        # set the time interval
        first_t_idx, last_t_idx = self.Model._get_t_indexes(t0,tf)
        # limit the data in this interval
        tbins = tbins[first_t_idx:last_t_idx]
        rates2D = {k:v[first_t_idx:last_t_idx,:] for k,v in rates2D.items()}

        N_E, N_t = len(Ebins), len(tbins)
        rates_E = dict()
        rates_t = dict()
        for nu in rates2D.keys():
            # rates_E[nu] = np.array([np.trapz(rates2D[nu][:,i], tbins) for i in range(N_E)])
            # rates_t[nu] = np.array([np.trapz(rates2D[nu][i,:], Ebins) for i in range(N_t)])
            rates_E[nu] = np.sum(rates2D[nu], axis=0)
            rates_t[nu] = np.sum(rates2D[nu], axis=1)
            
        rates_E['Total'], rates_t['Total'] = 0, 0
        for nu in rates2D.keys():
            rates_E['Total'] += rates_E[nu]
            rates_t['Total'] += rates_t[nu]

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=figsize)
        plt.suptitle(f'Between {tbins[0]:.1f} s and {tbins[-1]:.1f} s', y=1.02)
        ax1.set_xlabel('Recoil Energies [keV]', fontdict=font_small)
        ax2.set_xlabel('Times [s]', fontdict=font_small)
        ax1.set_ylabel('counts/keV/s', fontdict=font_small);
        ax2.set_ylabel('counts/keV/s', fontdict=font_small)
        for nu in rates_E.keys():
            ax1.plot(Ebins, rates_E[nu], label=nu)
        for nu in rates_t.keys():
            ax2.semilogx(tbins, rates_t[nu], label=nu)
        ax1.legend(fontsize=13); ax2.legend(fontsize=13);
        return fig, (ax1,ax2)

    def plot_recoil_spectra2D(self, integrated=False, t0=None, tf=None, figsize=None):
        """
        Plot 2D Recoil spectrum.
        Number of interactions at the detector as functions of 
        time and recoil energy
        Recoil energy part is same as 1D spectrum `plot_recoil_spectra`
        Returns 1D versions, and a density map.
        """
        figsize = figsize or (15,15)
        try : 
            rates2D = self.Model.rates2D
        except: 
            return print('2D rates do not exist!')
        if integrated: return self._integrated_spectra2D(t0,tf,figsize)
        tbins = self.Model.t 
        Ebins = self.Model.recoil_en

        fig, axes = plt.subplots(ncols=3, nrows=3, figsize=figsize)
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        for j, nu in enumerate(rates2D.keys()):
            for i in range(len(tbins[::10])):
                axes[j,0].plot(Ebins, rates2D[nu][i,:])
            axes[j,0].set_ylabel(f'counts {nu}')
            axes[j,0].set_xlabel(f'Recoil Energy [keV]')

            for i in range(len(Ebins)):
                axes[j,1].semilogx(tbins, rates2D[nu][:,i])
            axes[j,1].set_xlabel(f'Time [s]')

            axes[j,2].imshow(rates2D[nu].T, extent=(np.amin(Ebins), np.amax(Ebins),
                                                    np.amin(tbins), np.amax(tbins)),
                       cmap='jet', aspect='auto', origin='lower') #, clim=(0.0,3)) # , norm=LogNorm()
            axes[j,2].set_ylabel(f'Recoil Energy [keV]')
            axes[j,2].set_xlabel(f'Times [s]')
            for i in [0,1,2]:
                axes[j,i].tick_params(axis='both', which='major', labelsize=12)
                # if i == 0 or i == 1:
                #     continue
                # axes[j,i].set_xscale('log')
        return fig, axes

    def plot_recoil_spectra3D(self, total=True, figsize=None):
        """
        Plots the 2D recoil spectra on a 3D grid
        invoke %matplotlib notebook
        to play with the graph
        """
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from matplotlib.ticker import LinearLocator

        tbins, Ebins = self.Model.t, self.Model.recoil_en  
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10,3))
        X, Y = np.meshgrid(Ebins, np.log10(tbins))
        if total:
            Z = self.Model.total_rate2D
            # Z[Z==0] = np.nan
            surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, linewidth=0, antialiased=False)
        else:
            import matplotlib.patches as mpatches
            rates2D = self.Model.rates2D
            patches = []
            for key, cmap, c in zip(rates2D.keys(), 
                                    [cm.Blues,cm.Greens, cm.Reds],
                                    ['blue','green','red']):
                Z = rates2D[key]
                surf = ax.plot_surface(X, Y, Z, cmap=cmap, linewidth=0, antialiased=False, 
                                       alpha=0.75, label=key)
                patches.append(mpatches.Patch(color=c, label=key))

            ax.legend(handles=patches, fontsize=15)

        ax.zaxis.set_major_locator(LinearLocator(3))
        ax.set_xlabel('Recoil Energy [keV]'); ax.set_ylabel('log(Time) [s]'); ax.set_zlabel('Flux cts')
        plt.show()
        return fig, ax