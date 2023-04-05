import argparse
from multimessenger.supernova import Supernova_Models
from multimessenger.supernova.interactions import Interactions
import numpy as np
import cutax
import straxen


parser = argparse.ArgumentParser(
    description=('Script to run a SN simulation of a given model.')
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

def fetch_context():
    """ If context is updated, change it in here
    """
    return cutax.contexts.xenonnt_sim_SR0v4_cmt_v9(output_folder="/project2/lgrandi/xenonnt/simulations/supernova/")


def main():
    straxen.print_versions("strax,straxen,cutax,wfsim".split(","))
    SelectedModel = get_model(config_file, model_name, model_index, distance)
    Interaction = get_interactions(SelectedModel, distance, volume)
    context = fetch_context()
    # simulate
    Interaction.simulate_automatically(context=context, runid=runid)

if __name__ == "__main__":
    main()
