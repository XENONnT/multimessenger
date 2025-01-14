
try:
    from ._version import version as __version__
except ImportError:
    pass

import configparser
import os
if not os.path.exists("./snax_data/"):
    # Create the folder if it doesn't exist
    os.makedirs("snax_data")

# midway cluster default location
default_base_path = "/project2/lgrandi/xenonnt/simulations/supernova"
alternative_base = "./snax_data/"
if not os.path.exists(default_base_path):
    # Create the config parser
    config = configparser.ConfigParser()

    # Set the paths in the config file
    config['paths'] = {
        'base': alternative_base,
        'snewpy_models': './SNEWPY_models/',
        'processed_data': alternative_base,
        'imgs': alternative_base,
        'data': alternative_base,
        'outputs': alternative_base
    }

    config['wfsim'] = {
        'sim_folder': alternative_base,
        'instruction_path': alternative_base,
        'logs': alternative_base
    }

    # Save the config file, no need to check if it exists
    # it is called temp_config.ini and will be overwritten
    with open('temp_config.ini', 'w') as configfile:
        config.write(configfile)
