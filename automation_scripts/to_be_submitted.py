
import configparser
import straxen
import numpy as np
import os, sys
straxen.print_versions("strax,straxen,cutax,wfsim,snax,pema".split(","))
from snax.Supernova_Models import Models
from snax.sn_utils import fetch_context
from snax.interactions import Interactions
from snewpy.models.ccsn import Bollig_2016, Fornax_2021, Nakazato_2013

snewpy_models_path = "/project2/lgrandi/xenonnt/simulations/supernova/SNEWPY_models/"

# number of simulations
N = 20

case = sys.argv[1]
try:
    index = sys.argv[2]
except IndexError:
    index = 0

if case == 'N13':
    _N13_m13 = Nakazato_2013(filename=snewpy_models_path+"Nakazato_2013/nakazato-shen-z0.004-t_rev100ms-s13.0.fits")
    N13_m13 = Models(_N13_m13, save_name='N13_m13')
    N13_m13.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200), force=False)
    N13_m13.scale_fluxes(distance=10)
    Int_N13 = Interactions(N13_m13, Nuclei='Xenon', isotope='mix')                          # create interactions
    ############################################
    _N13_m50 = Nakazato_2013(filename=snewpy_models_path+"Nakazato_2013/nakazato-shen-z0.004-t_rev100ms-s50.0.fits")
    N13_m50 = Models(_N13_m50, save_name='N13_m50')
    N13_m50.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200), force=False)
    N13_m50.scale_fluxes(distance=10)
    Int_N50 = Interactions(N13_m50, Nuclei='Xenon', isotope='mix')                          # create interactions
    ############################################
    _N13_BHshen = Nakazato_2013(filename=snewpy_models_path+"Nakazato_2013/nakazato-shen-BH-z0.004-s30.0.fits")
    N13_BHshen = Models(_N13_BHshen, save_name='N13_BHshen')
    N13_BHshen.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200), force=False)
    N13_BHshen.scale_fluxes(distance=10)
    Int_BHshen = Interactions(N13_BHshen, Nuclei='Xenon', isotope='mix')                          # create interactions
    ############################################
    _N13_BHLS220 = Nakazato_2013(filename=snewpy_models_path+"Nakazato_2013/nakazato-LS220-BH-z0.004-s30.0.fits")
    N13_BHLS220 = Models(_N13_BHLS220, save_name='N13_BHLS220')
    N13_BHLS220.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200), force=False)
    N13_BHLS220.scale_fluxes(distance=10)
    Int_BHLS220 = Interactions(N13_BHLS220, Nuclei='Xenon', isotope='mix')                          # create interactions
    interactions = [Int_N13, Int_N50, Int_BHshen, Int_BHLS220]
elif case == 'B16':
    _B16_m11 = Bollig_2016(filename=snewpy_models_path + "Bollig_2016/s11.2c")
    B16_m11 = Models(_B16_m11, save_name='B16_m11')
    B16_m11.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200))
    B16_m11.scale_fluxes(distance=10)
    Int_B11 = Interactions(B16_m11, Nuclei='Xenon', isotope='mix')  # create interactions
    #####################################
    _B16_m27 = Bollig_2016(filename=snewpy_models_path + "Bollig_2016/s27.0c")
    B16_m27 = Models(_B16_m27, save_name='B16_m27')
    B16_m27.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200))
    B16_m27.scale_fluxes(distance=10)
    Int_B27 = Interactions(B16_m27, Nuclei='Xenon', isotope='mix')  # create interactions
    interactions = [Int_B11, Int_B27]
elif case == 'F21':
    _F21_m13 = Fornax_2021(filename=snewpy_models_path + "Fornax_2021/lum_spec_13M_r10000_dat.h5")
    F21_m13 = Models(_F21_m13, save_name='F21_m13')
    F21_m13.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200))
    F21_m13.scale_fluxes(distance=10)
    Int_F13 = Interactions(F21_m13, Nuclei='Xenon', isotope='mix')  # create interactions
    Int_F13.compute_interaction_rates()  # compute rates
    Int_F13.scale_rates(distance=10, volume=5.9)  # scale rates for dist & vol
    ###############################################
    _F21_m27 = Fornax_2021(filename=snewpy_models_path + "Fornax_2021/lum_spec_26.99M_r10000_dat.h5")
    F21_m27 = Models(_F21_m27, save_name='F21_m27')
    F21_m27.compute_model_fluxes(neutrino_energies=np.linspace(0, 30, 200))
    F21_m27.scale_fluxes(distance=10)
    Int_F27 = Interactions(F21_m27, Nuclei='Xenon', isotope='mix')  # create interactions
    interactions = [Int_F13, Int_F27]
else:
    raise ValueError('Case not recognized')

if len(interactions) <= index:
    interaction = interactions[index]
else:
    raise ValueError('Index out of range')

interaction.compute_interaction_rates()  # compute rates
interaction.scale_rates(distance=10, volume=5.9)

# get the data
distance = interaction.distance.value
object_name = interaction.Model.object_name.split('.')[0]
runid = f"{object_name}_{distance}kpc"

_conf = configparser.ConfigParser()
_conf.read("../simple_config.conf")
context = fetch_context(_conf)

def main():
    for i in range(N):
        truth_exists = context.is_stored(f"{runid}_{i:03}", "truth")
        peak_basics_exists = context.is_stored(f"{runid}_{i:03}", "peak_basics")
        if truth_exists and peak_basics_exists:
            print(f"Already simulated {runid}_{i:03}, skipping")
            continue
        interaction.simulate_automatically(runid=f"{runid}_{i:03}", context=context)
        print(f"\t\t simulated {runid}_{i:03}  ####")
        # After all created, remove low level data
        # Higher level should still be created
        # see https://straxen.readthedocs.io/en/latest/reference/datastructure_nT.html
        outpath = os.path.join(_conf["wfsim"]["sim_folder"], "strax_data")
        print(f"Data simulated, deleting the intermediate products.")
        for dtype in ["*lone_hits*", "*merged_s2s*", "*peaklet*", "*pulse*", "*raw*"]:
            files = os.path.join(outpath,  dtype)
            os.system(f'rm -r {files}')
        print(f"\t\t ##### {runid}_{i:03} COMPLETED ####\n")

if __name__ == "__main__":
    main()
