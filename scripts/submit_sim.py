import argparse
import os
from multimessenger.supernova.snewpy_models import models_list
import configparser

parser = argparse.ArgumentParser(
    description=('Script to submit request a simulation of a given SN '
                 'model via batch job.')
)

parser.add_argument('-c', '--config',
                    help=('The config file with the data paths'),
                    type=str, required=True)
parser.add_argument('-m', '--model',
                    help='Model to simulate, call --help for list of models.',
                    choices = models_list, type=str, required=True)
parser.add_argument('-i', '--model_index',
                    help=('Some model types have different options. Select them by'
                          'index. TO BE IMPROVED.'),
                    type=str, required=True)
parser.add_argument('-rib', '--runidbase', help='runid base.', type=str, required=True)
parser.add_argument('-N', '--ntotal', help='Number of realizations to simulate. For N=M, simulates the model M times.',
                    type=int, default=1, required=False)
parser.add_argument('-d', '--distance', help='Distance to SN in kpc.', default = 10, type=float)
parser.add_argument('-v', '--volume', help='Volume in tons.', default = 5.9, type=float)


args = parser.parse_args()

config_file = args.config
model_name = args.model
model_index = args.model_index
ntotal = args.ntotal
distance = args.distance
volume = args.volume
runidbase = args.runidbase

def make_batch_script(config, model_name, model_index, distance, volume, runid, ntotal):
    _conf = configparser.ConfigParser()
    _conf.read(config)
    outpath = _conf["wfsim"]["sim_folder"]

    main_str = f"""#!/bin/bash
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t
#SBATCH --job-name={runid}_{ntotal}
#SBATCH --output={outpath}{runid}_{ntotal}.out
#SBATCH --error={outpath}{runid}_{ntotal}.err
#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=12:00:00
module load singularity
singularity shell \\
    --bind /cvmfs/ \\
    --bind /project/ \\
    --bind /project2/ \\
    --bind /scratch/midway2/$USER \\
    --bind /dali \\
    /project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg <<EOF
python simulate_snmodel.py -c {config} -m {model_name} -i {model_index} -d {distance} -v {volume} -id {runid} -N {ntotal}
EOF"""
    with open(f'SN_{runid}_{ntotal}.job', 'w') as F:
        F.write(main_str)  
    print(f'Generated file with ID: {runid}_{ntotal}')
    

if __name__ == "__main__":
    print('Making .job file:')
    make_batch_script(config_file, model_name, model_index, distance, volume, runidbase, ntotal)
    os.system(f"sbatch SN_{runidbase}_{ntotal}.job")

    # for i in range(ntotal):
    #     incr = f"{i:03}"
    #     make_batch_script(config_file, model_name, model_index, distance, volume, runidbase, incr)
    #     os.system(f"sbatch SNsim_{runidbase}_{incr}.job")