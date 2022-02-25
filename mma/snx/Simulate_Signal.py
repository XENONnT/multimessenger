#!/usr/bin/python

## The very first version
## Absolutely, hideous. Gonna work on it
##
from .constants import *
from .libraries import *

# these mean we don't compute photon times. 
# it's a bit more crude but over 100x faster. 
# So always do for a first-order approximation. 
use_timing = -1
output_timing = 0
wftime = [0, 0, 0]
wfamp = [0., 0., 0.]

class vectorize(np.vectorize):
    def __get__(self, obj, objtype):
        return functools.partial(self.__call__, obj)

class Simulator:
    """ Simulate signal based on Recoil Energy rates
        It uses 
    """
    def __init__(self, name, detector=None):
        self.name = name
        self.detector = detector or nestpy.DetectorExample_XENON10 #LUX_RUN3
        self.nc = nestpy.NESTcalc(self.detector())
        self.A = 131.293 # avg atomic mass of Xe
        self.Z = 54. # Atomic number of Xe 
        self.density = 2.85 # 2.9  # LXe density; g/cm^3
        self.g2_params = self.nc.CalculateG2(False)
        self.g2 = self.g2_params[3]
        
    def _inverse_transform_sampling(self, x_vals, y_vals, n_samples):
        """ Strangely, it does not take self automatically
            Needs to be called as
            a = Simulator
            a._inverse_transform_sampling(a, x, y, N)

        """
        cum_values = np.zeros(x_vals.shape)
        y_mid = (y_vals[1:]+y_vals[:-1])*0.5
        cum_values[1:] = np.cumsum(y_mid*np.diff(x_vals))
        inv_cdf = itp.interp1d(cum_values/np.max(cum_values), x_vals)
        r = np.random.rand(n_samples)
        return inv_cdf(r)

    @vectorize
    def Get_LyQy(self, interaction=nestpy.NR, 
                  energy=100., # energy in keV of the recoil itself
                  drift_field=81.):
        y = self.nc.GetYields(interaction, energy, self.density, drift_field, self.A, self.Z)
        return y.PhotonYield, y.ElectronYield

    @vectorize
    def Get_quanta(self,interaction=nestpy.NR, 
                  energy=100., # energy in keV of the recoil itself
                  drift_field=81.):
        y = self.nc.GetYields(interaction, energy, self.density, drift_field, self.A, self.Z)
        q = self.nc.GetQuanta(y, self.density)
        n_p, n_e = q.photons, q.electrons
        return n_p, n_e

    def simulate_quanta(self, Er_sampled, drift_field=200, plot=False, 
                        figsize=(6,6), mono_energetic=False):
        """ simulate quanta based on sampled energies
            Er_sampled : `array`
                The sampled Energues
            drift_field : `float`, optional
                Electric field drifting electrons V/cm
            plot : `bool`, optional
                whether to plot the output
            figsize : `tuple`, optional
                figure size, default is (6,6)
            mono_energetic : `bool`, optional
                Whether the signal is monoenergetic. If true, adds a nestpy
                prediction in the plot. (Er_sampled is expected to be repeating)
        """
        Ly_j, Qy_j = self.Get_LyQy(energy=Er_sampled, drift_field=drift_field)
        n_p, n_e = self.Get_quanta(energy=Er_sampled, drift_field=drift_field)
        if plot:
            fig, ax = plt.subplots(ncols=1, figsize=figsize)
            ax.hist2d(n_p, n_e, bins=50, cmin=1, norm=LogNorm(), cmap ='plasma')
            ax.scatter(Ly_j[0], Qy_j[0], color='g', marker='+', s=500)
            if mono_energetic:
                ax.plot(Ly_j*Er_sampled, Qy_j*Er_sampled, c='red', alpha=0.5, label='Nestpy Prediction')
            ax.set_xlabel(r'n$_\gamma$', fontdict=font_small)
            ax.set_ylabel(r'n$_e$', fontdict=font_small)
            ax.set_title(f'Nestpy for ({drift_field} V/cm)', fontdict=font_small)
            ax.legend(fontsize='large')
        return n_p, n_e
            
    @vectorize
    def get_drift_v(self, field, temp = 177.15):
        return self.nc.SetDriftVelocity(temp, self.density, field) # temp in K (kelvin)


    @vectorize
    def GetS1_S2(self, interaction = nestpy.NR, 
                  energy = 100.,
                  drift_field = 200): 
            """ This function gets both S1 and S2 for reliable anticorrelation.
            Parameters
            ----------
            interaction : `nestpy.interaction`, optional
                The type of interaction. Nuclear Recoil by default.
            energy : `float`, optional
                Energy in keV of the Recoil. Default is 100
            drift_field : `float`, optional
                Electric drift field, default os 200 V/cm

            Returns
            -------
            `tuple` : (cs1, cs2) corrected S1 and S2 areas
            
        """
        y = self.nc.GetYields(interaction, energy, self.density, drift_field, self.A, self.Z) 
        q = self.nc.GetQuanta(y, self.density)

        driftv = self.get_drift_v(drift_field)
        dv_mid = self.get_drift_v(drift_field) #assume uniform field 
        maxposz = 560 # max z position in mm

        #We'll randomly select some positions to get realistic xyz smearing. 
        # R = 200 mm
        random_r = np.sqrt(np.random.uniform(0, 200.**2)) 
        random_theta = np.random.uniform(0, 2*np.pi)

        truthposx, truthposy, truthposz = random_r*np.cos(random_theta), random_r*np.sin(random_theta),np.random.uniform(60., 540.)
        smearposx, smearposy, smearposz = truthposx, truthposy, truthposz # random positions?

        dt = np.abs((truthposz - maxposz)/driftv)
        S1 = self.nc.GetS1(q, truthposx, truthposy, truthposz,
                            smearposx, smearposy, smearposz,
                            driftv, dv_mid, # drift velocities (assume homogeneous drift field)
                            interaction, 1, # int type, event # 
                            drift_field, energy, #dfield, energy
                            use_timing, output_timing,
                            wftime,
                            wfamp)

        S2 = self.nc.GetS2(q.electrons, 
                        truthposx, truthposy, truthposz,
                        smearposx, smearposy, smearposz,
                        dt, driftv, 1, #evt num
                        drift_field, use_timing, output_timing, # dv dvmid 
                        wftime, wfamp, 
                        self.g2_params)

        # S1[7] is the spike area, corrected for xyz position, and dividing out dpe effect.
        # S2[7] here is cs2_b, again correcting for DPE, xyz position, DAQ smearing
        cs1 = S1[2] # 7
        cs2 = S2[4] # 7
        return cs1, cs2

    def Plot_S1S2(self, energies, no_thr=True):
        S1, S2 = self.GetS1_S2(energy=energies)
        fig, ax = plt.subplots(ncols=1, figsize=(5, 5))
        cax = fig.add_axes([0.95, 0.15, 0.07, 0.7])
        if no_thr:
            m1 = S1 > 0
            m2 = S2 > 0
            ax.hist2d(np.abs(S1[~(m1 & m2)]), np.abs(S2[~(m1 & m2)]), bins=150, cmin=1, norm=LogNorm(), cmap='Greys')
            *_, im = ax.hist2d(S1[m1 & m2], S2[m1 & m2], bins=150, cmin=1, norm=LogNorm(), cmap='jet')
        else:
            *_, im = ax.hist2d(S1, S2, bins=150, cmin=1, norm=LogNorm(), cmap='jet')
        ax.set_xlabel(r'$S1$ [PE]', fontdict=font_small)
        ax.set_ylabel(r'$S2$ [PE]', fontdict=font_small)
        fig.colorbar(im, cax=cax, orientation='vertical')
        return ax