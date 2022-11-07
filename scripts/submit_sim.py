import argparse
import os
#from multimessenger.supernova import Supernova_Models as sn
models_list = [
 'Bollig_2016',
 'Fornax_2019',
 'Fornax_2021',
 'Kuroda_2020',
 'Nakazato_2013',
 'OConnor_2013',
 'OConnor_2015',
 'Sukhbold_2015',
 'Tamborra_2014',
 'Walk_2018',
 'Walk_2019',
 'Warren_2020',
 'Zha_2021']

parser = argparse.ArgumentParser(
    description=('Script to submit request a simulation of a given SN '
                 'model via batch job.')
)
parser.add_argument('-m', '--model',
                    help='Model to simulate, call --help for list of models.',
                    choices = models_list,
                    type=str,
                    required=True)
parser.add_argument('-i', '--model_index',
                    help=('Some model types have different options. Select them by'
                          'index. TO BE IMPROVED.'),
                    type=str,
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

model_name = args.model
model_index = args.model_index
ntotal = args.ntotal
distance = args.distance
volume = args.volume
runid = args.runid

def make_batch_script(model_name,model_index,ntotal,distance,volume,runid):
    main_str = f'''#!/bin/bash
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t
#SBATCH --job-name=wfsim_{runid}
#SBATCH --output=/dali/lgrandi/melih/mma/sim_logs/wfsim_{runid}.out
#SBATCH --error=/dali/lgrandi/melih/mma/sim_logs/wfsim_{runid}.err
#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=12:00:00
module load singularity
singularity shell \\
    --bind /cvmfs/ \\
    --bind /project2/ \\
    --bind /scratch/midway2/$USER \\
    --bind /dali \\
    /project2/lgrandi/xenonnt/singularity-images/xenonnt-2022.09.1.simg <<EOF
python simulate_snmodel.py -m {model_name} -i {model_index} -N {ntotal} -d {distance} -v {volume} -id {runid}
EOF''' 
    with open(f'SN_sim_{runid}_wfsim.job', 'w') as F:
        F.write(main_str)  
    print(f'Generated file with ID: {runid}')
    

if __name__ == "__main__":
    print('Making .job file:')
    make_batch_script(model_name,model_index,ntotal,distance,volume,runid)
    print('Launching batch job:')
    os.system(f"sbatch SN_sim_{runid}_wfsim.job")
