
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u
import pandas as pd
import plotly.express as px
from snewpy.neutrino import Flavor
from .sn_utils import isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

plt.style.use('ggplot')

plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'

plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['xtick.major.pad'] = 5
plt.rcParams['ytick.major.pad'] = 5

plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['xtick.minor.pad'] = 5
plt.rcParams['ytick.minor.pad'] = 5
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['font.size'] = 16

font_small = {'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 16,
              }

font_medium = {'family': 'serif',
               'color': 'darkred',
               'weight': 'normal',
               'size': 20,
               }

font_large = {'family': 'serif',
              'color': 'darkred',
              'weight': 'normal',
              'size': 24,
              }
params = {'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          }
# Updates plots to apply the above formatting to all plots in doc
plt.rcParams.update(params)


class Plotter:
    """ Plotter Class for SN data
    """

    def __init__(self,
                 model,
                 figsize=(17, 4),
                 cmap='jet'):
        """
        """
        self.model = model
        self.figsize = figsize
        self.cmap = matplotlib.cm.get_cmap(cmap)
        self.colors = ['r', 'g', 'b', 'orange', 'teal', 'brown', 'gold']
        self.__version__ = "0.1.2"

    def plot_recoil_spectrum(self, isotopes=False,
                             distance=10*u.kpc):
        rateEr, ratet = self.model.scale_rates(isotopes=isotopes,
                                               distance=distance,
                                               overwrite=False)
        if rateEr is None or ratet is None:
            raise TypeError("Rates are not calculated! Run <obj>.compute_rates()")

        # plotly-fy it by creating a df
        _rateErtotal = rateEr["Total"]
        _ratettotal = ratet["Total"]
        rateEr = {f.to_tex(): rateEr[f] for f in Flavor}
        ratet = {f.to_tex(): ratet[f] for f in Flavor}
        rateEr["Total"] = _rateErtotal
        ratet["Total"] = _ratettotal
        rateEr["Recoil Energies"] = self.model.recoil_energies
        ratet["Time"] = self.model.model.time
        er_unit = str(_rateErtotal.unit)
        t_unit = str(_ratettotal.unit)

        df_Er = pd.DataFrame.from_dict(rateEr)
        df_t = pd.DataFrame.from_dict(ratet)
        fig = px.line(df_Er, x="Recoil Energies", y=df_Er.columns[:-1], title=f"Recoil spectrum at {distance}")
        fig.update_layout(yaxis_title=f'Rates [{er_unit}]',
                          xaxis_title=f"Recoil Energies [{str(self.model.recoil_energies.unit)}]")
        fig.show()
        fig = px.line(df_t, x="Time", y=df_t.columns[:-1], title=f"Recoil spectrum at {distance}")
        fig.update_layout(yaxis_title=f'Rates [{t_unit}]',
                          xaxis_title=f"Time [{str(self.model.model.time.unit)}]")
        fig.show()


    def plot_form_factor(self):
        Er_tests = np.linspace(0, 300, 1000) * u.keV
        cmap = plt.cm.jet
        for i, nuclide in enumerate(self.model.Nucleus):
            ffacts = nuclide.form_factor(Er_tests) ** 2
            plt.semilogy(Er_tests, ffacts, label=f'{nuclide.name}', color=cmap(i / len(self.model.Nucleus)))
        plt.legend(fontsize=13)
        plt.ylabel(f'Helm Form Factor $F(E_r)^2$', fontsize=17)
        plt.xlabel(r'$E_r$ [keV]', fontsize=17)

    def plot_cross_section(self, neutrino_energies=np.linspace(0, 100, 100) * u.MeV,
                           recoil_energies=np.linspace(0, 15, 150) * u.keV):
        cs_test = self.model.Nucleus[0].nN_cross_section(neutrino_energies, recoil_energies)
        cb = plt.pcolormesh(cs_test * 1e43, cmap='PuBu')  # ,norm=matplotlib.colors.LogNorm())
        plt.xlabel(r'E$_\nu$ [MeV]', fontsize=16)
        plt.ylabel(r'E$_r$ [keV]', fontsize=16)
        cbb = plt.colorbar(cb)
        cbb.set_label('cross-section\n'
                      fr'[x$10^{{43}}$ {cs_test.unit}]', fontsize=16)
        plt.xticks(ticks=np.arange(0, 100, 10))
        plt.yticks(ticks=np.arange(0, 150, 10), labels=np.arange(0, 15, 1))
        return plt.gcf(), plt.gca()

    def plot_cross_section_fantastic(self, neutrino_energies=np.linspace(0, 100, 100) * u.MeV,
                                     recoil_energies=np.linspace(0, 15, 150) * u.keV):
        fig, axes = plt.subplots(ncols=len(self.model.Nucleus), figsize=(len(self.model.Nucleus)*4, 4))
        for i, n in enumerate(self.model.Nucleus):
            cs_test = n.nN_cross_section(neutrino_energies, recoil_energies)
            cb = axes[i].pcolormesh(cs_test * 1e43, cmap='PuBu')  # ,norm=matplotlib.colors.LogNorm())
            axes[i].set_xlabel(r'E$_\nu$ [MeV]', fontsize=16)
            axes[i].set_ylabel(r'E$_r$ [keV]', fontsize=16)
            axes[i].set_title(n.name)
            # cbb = plt.colorbar(cb)
            # cbb.set_label('cross-section\n'
            #               fr'[x$10^{{43}}$ {cs_test.unit}]', fontsize=16)
            axes[i].set_xticks(ticks=np.arange(0, 100, 10))
            axes[i].set_yticks(ticks=np.arange(0, 150, 10), labels=np.arange(0, 15, 1))


    def plot_mean_cross_section(self, er_vals=np.linspace(0, 20, 150)*u.keV):
        xsec_t = {f: np.zeros((len(er_vals), len(self.model.model.time))) for f in Flavor}

        for f in Flavor:
            xsec_t[f] = self.model.Nucleus[0].nN_cross_section(self.model.model.meanE[f], er_vals)

        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(20, 4))
        plt.subplots_adjust(wspace=0.45, hspace=0.3)
        plt.suptitle("Mean Cross Section at different times", y=0.98)
        for j, f in enumerate(Flavor):
            im = axes[j].imshow(xsec_t[f], extent=(np.amin(self.model.model.time.value), np.amax(self.model.model.time.value),
                                                   np.amin(er_vals.value), np.amax(er_vals.value),),
                                cmap='jet', aspect='auto', origin='lower')
            divider = make_axes_locatable(axes[j])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical', label=f'cross-section {xsec_t[Flavor.NU_E].unit}')
            axes[j].set_title(f.to_tex())
            axes[j].set_xlabel(f'Times [{self.model.model.time.unit}]')
            axes[j].tick_params(axis='both', which='major', labelsize=12)
        axes[0].set_ylabel(f'Recoil Energy [{er_vals.unit}]')
        return fig, axes

    def plot_params(self):
        """ plot the model parameters.
            luminosity, mean energies, and alpha pinch parameter
        """
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 5))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        for flavor in Flavor:
            kwargs = dict(label=flavor.to_tex(), color='C0' if flavor.is_electron else 'C1',
                          ls='-' if flavor.is_neutrino else ':', lw=2, alpha=0.7)
            ax1.plot(self.model.model.time, self.model.model.luminosity[flavor] / 1e51, **kwargs)
            ax2.plot(self.model.model.time, self.model.model.meanE[flavor], **kwargs)
            ax3.plot(self.model.model.time, self.model.model.pinch[flavor], **kwargs)

        for a in [ax1, ax2, ax3]:
            a.set_xscale('log')

        ax3.legend(loc='upper right', ncol=2, fontsize=16)
        ax1.set(ylabel='luminosity [foe s$^{-1}$]', xlabel='time [s]')
        ax2.set(ylabel=r'mean $\nu$ E [MeV]', xlabel='time [s]')
        ax3.set(ylabel=r'$\alpha$', xlabel='time [s]')

    def plot_flux(self, time=50*u.ms, neutrino_energy=100*u.MeV):
        ispec_t = self.model.model.get_initial_spectra(time, self.model.neutrino_energies) # init spect, oscillated possible
        fluxunit = ispec_t[Flavor.NU_E].unit

        # construct a dictionary containing each {flavorname-array} pairs for each flavor
        ispec_E = {f: np.zeros(len(self.model.model.time)) * fluxunit for f in Flavor}
        for i, t in tqdm(enumerate(self.model.model.time), total=len(self.model.model.time)):
            temp = self.model.model.get_initial_spectra(t, neutrino_energy)
            for f in Flavor:
                ispec_E[f][i] = temp[f]

        neutrino_energies = self.model.neutrino_energies
        times = self.model.model.time
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
        plt.subplots_adjust(wspace=0.25, hspace=0.3)
        for f in Flavor:
            kwargs = dict(label=f.to_tex(), color='C0' if f.is_electron else 'C1',
                          ls='-' if f.is_neutrino else ':', lw=2, alpha=0.7)
            ax1.plot(neutrino_energies, ispec_t[f], **kwargs)
            ax2.plot(times, ispec_E[f], **kwargs)

        try:
            ax1.set(xlabel=fr'$E$ [{neutrino_energies.unit}]', title=fr'{self.model.model.EOS}: '
                                                                     fr'{self.model.model.progenitor_mass.value} $M_\odot$\ ' 
                                                                     fr' Initial Spectra: t=50 ms',
                    ylabel=fr'flux [{ispec_t[Flavor.NU_E].unit}]')
            ax2.set(xlabel=fr'$t$ [{times.unit}]', title=fr'{self.model.model.EOS}: '
                                                         fr'{self.model.model.progenitor_mass.value} $M_\odot$\ '
                                                         'Initial Spectra: E=10 MeV',
                    ylabel=fr'flux [{ispec_E[Flavor.NU_E].unit}]')
        except:
            pass
        ax1.grid(True); ax2.grid(True)
        ax2.set_xscale('log')
        ax2.legend(loc='upper right', ncol=2, fontsize=16)


    def plot_sampled_data(self, n, bins=100, dtype='energy', xscale='log'):
        data, _xaxis, _yaxis = self.model.sample_data(n, dtype=dtype, return_xy=True)
        xaxis = _xaxis.value
        yaxis = _yaxis.value
        # Er_sampled = self.Model.sample_from_recoil_spectrum(x=x.lower(), N_sample=N_sample)
        E_thr_tot = np.trapz(yaxis, xaxis)
        pdf_diff_rate = yaxis / E_thr_tot
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(xaxis, pdf_diff_rate, label='pdf', lw=3, alpha=0.7)
        if xscale == 'log':
            bins = np.logspace(np.log10(xaxis[xaxis > 0].min()), np.log10(xaxis.max()), bins)
            ax.set_xscale('log')
        ax.hist(data, histtype='step', lw=3, bins=bins, alpha=1, density=True, label='Sampled Points')
        ax.legend(fontsize=16)

        xl = f'Time [{_xaxis.unit}]' if type == 'time' else f'Recoil Energy [{_xaxis.unit}]'
        ax.set_xlabel(xl)
        return data

    def plot_counts(self, volumes=np.linspace(3,50,10),
                    distances=np.linspace(2,60,10),
                    figsize=(10,10)):
        volumes = volumes * u.tonne
        distances = distances * u.kpc
        total_counts = np.zeros((len(distances), len(volumes))) * u.ct

        for i, d in enumerate(distances):
            rates_scaled_Er, _ = self.model.scale_rates(distance=d)
            for j, v in enumerate(volumes):
                total_counts[i, j] = np.trapz(rates_scaled_Er['Total'] * v, self.model.recoil_energies)

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f"Counts for {self.model.name}")
        ax.matshow(total_counts, cmap=plt.cm.Greens, origin='lower', norm=LogNorm())
        ax.set_xlabel("Volumes [t]", weight='bold')
        ax.xaxis.set_label_position('top')
        ax.set_ylabel("Distance [kpc]", weight='bold')

        for j, v in enumerate(volumes):
            ax.axvline(j - 0.5)
        for i, d in enumerate(distances):
            ax.axhline(i - 0.5)
            for j, v in enumerate(volumes):
                c = total_counts[i, j]
                ax.text(j, i, f"{int(c.value)}", va='center', ha='center', weight='bold')

        ax.set_xticks(np.arange(len(volumes)), np.round(volumes.value).astype(int))
        ax.set_yticks(np.arange(len(distances)), np.round(distances.value).astype(int))
        ax.set_aspect('auto')
        ax.grid(False)
