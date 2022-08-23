import argparse
from multimessenger.supernova import Supernova_Models as sn

parser = argparse.ArgumentParser(
    description=('Script to run a SN simulation of a given model.')
)
parser.add_argument('-m', '--model',
                    help='Model to simulate, call --help for list of models.',
                    choices = sn.models_list,
                    type=str,
                    required=True)
parser.add_argument('-i', '--model_index',
                    help=('Some model types have differnt options. Select them by'
                          'index. TO BE IMPROVED.'),
                    type=int,
                    required=True)
parser.add_argument('-N', '--ntotal',
                    help='Number of events to simulate. -1 for realistic simulation.',
                    type=int,
                    required=True)
parser.add_argument('-d', '--distance',
                    help='Distance to SN in kpc.',
                    default = 10,
                    type=float)
parser.add_argument('-id', '--runid',
                    help='runid to consider.',
                    type=str,
                    required=True)

args = parser.parse_args()

import os
import numpy as np

import nestpy
import pandas as pd
import straxen
import astropy.units as u
import cutax

model_name = args.model
model_index = args.model_index
ntotal = args.ntotal
distance = args.distance
runid = args.runid

def main():
    
    A = sn.Models(model_name=model_name, index=model_index)
    A.compute_rates(); #fetches the already existing sim
    
    _rate, _ = A.scale_rates(distance=distance*u.kpc)
    nevents = int(np.trapz(_rate['Total'] * 5.9*u.t, A.recol_energies).value)

    sampled_Er = A.sample_data(nevents)
    #sampled_t = A.sample_data(nevents, dtype='time')
    
    field_file="fieldmap_2D_B2d75n_C2d75n_G0d3p_A4d9p_T0d9n_PMTs1d3n_FSR0d65p_QPTFE_0d5n_0d4p.json.gz"
    field_map = straxen.InterpolatingMap(
                        straxen.get_resource(downloader.download_single(field_file),
                                            fmt="json.gz"),
                        method="RegularGridInterpolator")

    nc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())
    ## not sure if nestpy RNG issue was solved, so randomize NEST internal state
    for i in range(np.random.randint(100)):
        nc.GetQuanta(nc.GetYields(energy=np.random.uniform(10,100)))
    
    if ntotal == -1:
        N_events = nevents
    else:
        N_events = ntotal
        
    instr = A.generate_instructions(energy_deposition=sampled_Er, 
                                    timemode="shifted", 
                                    n_tot=N_events, 
                                    nc=nc, 
                                    fmap=field_map)
    df = pd.DataFrame(instr)
    print(f"Total duration {np.ptp(df['time'])*1e-9:.2f} seconds")
    
    st = cutax.contexts.xenonnt_sim_SR0v2_cmt_v8(cmt_run_id="026000", 
        output_folder=os.path.join(A.config['wfsim']['sim_folder'], "strax_data"))
    
    st = A.simulate_one(df, runid, context=st)
    
if __name__ == "__main__":
    main()
