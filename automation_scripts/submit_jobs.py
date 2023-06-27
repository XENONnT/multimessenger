import argparse
import os
import configparser


parser = argparse.ArgumentParser(
    description=('Script to submit request a simulation of a given SN '
                 'model via batch job.\n '
                 'Ex: python3 submit_sim.py -c ../simple_config.conf -m Bollig2016 -i 0 -N 20')
)

parser.add_argument('-c', '--config', help=('The config file with the data paths'), type=str, required=True)
parser.add_argument('-m', '--model', help='Model to simulate, call --help for list of models.',
                    choices=['Bollig2016', 'Nakazato2013', 'Fornax2021'], type=str, required=True)
parser.add_argument('-i', '--model_index', help=('Some model types have differnt options. Select them by' 'index.'), type=int, required=True)
parser.add_argument('-N', '--ntotal', help=('Number of realizations. By default it is 1.'), type=int, default=1, required=False)
parser.add_argument('-d', '--distance', help='Distance to SN in kpc.', default = 10, type=float)
parser.add_argument('-v', '--volume', help='Volume in tons.', default = 5.9, type=float)

args = parser.parse_args()

config_file = args.config
model_name = args.model
model_index = args.model_index
ntotal = args.ntotal
distance = args.distance
volume = args.volume

def make_batch_script(config, model_name, model_index, distance, volume, ntotal):
    _conf = configparser.ConfigParser()
    _conf.read(config)
    outpath = _conf["wfsim"]["sim_folder"]

    main_str = f"""#!/bin/bash
#SBATCH --qos=xenon1t
#SBATCH --partition=xenon1t
#SBATCH --job-name={model_name}_{model_index}
#SBATCH --output={outpath}logs/{model_name}_{model_index}_{ntotal}.out
#SBATCH --error={outpath}logs/{model_name}_{model_index}_{ntotal}.err
#SBATCH --account=pi-lgrandi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time=12:00:00
module load singularity
singularity shell \\
    --bind /project2/ \\
    --bind /scratch/midway2/$USER \\
    --bind /midway \\
    /project2/lgrandi/xenonnt/singularity-images/xenonnt-development.simg <<EOF
python to_be_submitted.py -c {config} -m {model_name} -i {model_index} -d {distance} -v {volume} -N {ntotal}
EOF"""
    with open(f'SN_{model_name}_{model_index}_{ntotal}.job', 'w') as F:
        F.write(main_str)
    print(f'Generated file with ID: SN_{model_name}_{model_index}_{ntotal}')


if __name__ == "__main__":
    print('Making .job file:')
    make_batch_script(config_file, model_name, model_index, distance, volume, ntotal)
    os.system(f"sbatch SN_{model_name}_{model_index}_{ntotal}.job")
