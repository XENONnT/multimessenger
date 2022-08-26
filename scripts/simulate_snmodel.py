import argparse
from multimessenger.supernova import Supernova_Models as sn
import numpy as np
import nestpy
import pandas as pd
import straxen
import astropy.units as u

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
parser.add_argument('-v', '--volume',
                    help='Volume in tons.',
                    default = 5.9,
                    type=float)
parser.add_argument('-id', '--runid',
                    help='runid to consider.',
                    type=str,
                    required=True)

args = parser.parse_args()

downloader = straxen.MongoDownloader()

model_name = args.model
model_index = args.model_index
ntotal = args.ntotal
distance = args.distance
volume = args.volume
runid = args.runid

def main():
    
    A = sn.Models(model_name=model_name, index=model_index, distance=distance*u.kpc, volume=volume*u.t)
    A.compute_rates() #fetches the already existing sim

    nevents = int(A.single_rate.value)
    
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
        sampled_t = A.sample_data(nevents, dtype='time')
    else:
        N_events = ntotal
        sampled_t = "shifted"
        
    sampled_Er = A.sample_data(N_events)
    instr = A.generate_instructions(energy_deposition=sampled_Er, 
                                    timemode=sampled_t, 
                                    n_tot=N_events, 
                                    nc=nc, 
                                    fmap=field_map)
    df = pd.DataFrame(instr)
    print(f"Total duration {np.ptp(df['time'])*1e-9:.2f} seconds")
    st = A.simulate_one(df, runid)
    
if __name__ == "__main__":
    main()
