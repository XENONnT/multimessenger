import argparse
import configparser

from snax import Supernova_Models
from multimessenger.supernova.interactions import Interactions
from multimessenger.supernova.sn_utils import fetch_context, make_json
import numpy as np
import straxen
import os


parser = argparse.ArgumentParser(
    description=('Script to run a SN simulation of a given model.'
                 'It allows for requesting multiple simulations at the same time.')
)

parser.add_argument('-c', '--config',
                    help=('The config file with the data paths'),
                    type=str,
                    required=True)
parser.add_argument('-m', '--model',
                    help='Model to simulate, call --help for list of models.',
                    choices = Supernova_Models.models_list,
                    type=str,
                    required=True)
parser.add_argument('-i', '--model_index',
                    help=('Some model types have differnt options. Select them by'
                          'index. TO BE IMPROVED.'),
                    type=int,
                    required=True)
parser.add_argument('-N', '--ntotal',
                    help=('Number of realizations. By default it is 1.'),
                    type=int,
                    default=1,
                    required=False)
parser.add_argument('-d', '--distance', help='Distance to SN in kpc.', default = 10, type=float)
parser.add_argument('-v', '--volume', help='Volume in tons.', default = 5.9, type=float)
parser.add_argument('-id', '--runid', help='runid to consider.', type=str, required=True)

args = parser.parse_args()

downloader = straxen.MongoDownloader()

number_of_realization = args.ntotal
config_file = args.config
model_name = args.model
model_index = args.model_index
distance = args.distance
volume = args.volume
runid = args.runid

def get_model(config_file, model_name, index, distance):
    SelectedModel = Supernova_Models.Models(model_name, config_file=config_file)
    SelectedModel(index=index)  # brings the attributes
    SelectedModel.compute_model_fluxes(neutrino_energies=np.linspace(0, 50, 200))
    _ = SelectedModel.scale_fluxes(distance=distance)
    return SelectedModel

def get_interactions(SelectedModel, distance, volume):
    Int = Interactions(SelectedModel, Nuclei='Xenon', isotope='mix')  # create interactions
    Int.compute_interaction_rates()  # compute rates
    _ = Int.scale_rates(distance=distance, volume=volume)  # scale rates for dist & vol
    return Int


def main():
    _conf = configparser.ConfigParser()
    _conf.read(config_file)

    straxen.print_versions("strax,straxen,cutax,wfsim".split(","))
    SelectedModel = get_model(config_file, model_name, model_index, distance)
    Interaction = get_interactions(SelectedModel, distance, volume)
    context = fetch_context(_conf)
    # simulate
    for realization in range(number_of_realization):
        try:
            Interaction.simulate_automatically(runid=f"{runid}_{realization:03}", context=context)
            # create a metadata for bookkeeping
            make_json(Interaction, f"{runid}_{realization:03}", config_file)
        except Exception as e:
            print(f"\n\n >>> Exception raised: for  < {runid}_{realization:03} >\n{e}\n\n")
        # simulates truth, peak basics, and peak positions

    # After all created, remove low level data
    # Higher level should still be created
    # see https://straxen.readthedocs.io/en/latest/reference/datastructure_nT.html
    outpath = os.path.join(_conf["wfsim"]["sim_folder"], "strax_data")
    print(f"Data simulated, deleting the intermediate products.")
    for dtype in ["*lone_hits*", "*merged_s2s*", "*peaklet*", "*pulse*", "*raw*"]:
        files = os.path.join(outpath,  dtype)
        os.system(f'rm -r {files}')

if __name__ == "__main__":
    main()
