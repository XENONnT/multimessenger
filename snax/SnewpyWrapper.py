import os, sys
import numpy as np
import pandas as pd
import snewpy
from .sn_utils import get_hash_from_model

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

snewpy_models_path = "/project2/lgrandi/xenonnt/simulations/supernova/SNEWPY_models/"


class SnewpyModel:
    """snewpy wrapper"""

    def __init__(self, local_models_path=None):
        # if manually given
        if local_models_path is not None:
            snewpy.model_path = local_models_path
        # check if we are on the cluster
        elif os.path.exists(snewpy_models_path):
            snewpy.model_path = snewpy_models_path
        # check if exists locally
        elif os.path.exists("./SNEWPY_models/"):
            snewpy.model_path = "./SNEWPY_models/"
        else:
            import warnings

            warnings.warn(
                "Not on the cluster, snewpy models are not available. "
                "Get models first \nimport snewpy; snewpy.get_models()"
            )
        from snewpy import _model_urls

        self.model_urls = _model_urls.model_urls
        self.model_keys = self.model_urls.keys()
        self.imported_snewpy_models = {}

    def _import_models(self, base, sntype="ccsn"):
        """Import the models if not already imported"""
        if base not in self.model_keys:
            print(f"Model {base} not available")
            sys.exit()

        if base in self.imported_snewpy_models:
            # already imported
            return None
        else:
            package = f"snewpy.models.{sntype}"
            model = getattr(__import__(package, fromlist=[base]), base)
            par_combinations = pd.DataFrame(model.get_param_combinations())
            # get hashes
            model_hashes = []
            for pars in par_combinations:
                _hash = get_hash_from_model(model(**pars))
                model_hashes.append(_hash)
            par_combinations["hash"] = model_hashes
            par_combinations["Combination Index"] = np.arange(len(par_combinations)) + 1
            par_combinations.set_index("Combination Index", inplace=True)
            self.imported_snewpy_models[base] = (model, par_combinations)
            return None

    def display_models(self, base=None, sntype="ccsn"):
        """Display the available models
        Get the model by calling `get_model('{base}', combination_index=#)`
        or by calling `get_model('{base}', **kwargs)`
        """
        if base is None:
            print(self.model_keys)
        else:
            self._import_models(base, sntype)
            print(
                f"\t Get the model by calling `get_model('{base}', combination_index=#)`\n"
                f"\t or by calling `get_model('{base}', **kwargs)`"
            )
            return self.imported_snewpy_models[base][1]

    def get_model(self, base, sntype="ccsn", combination_index=None, **kwargs):
        """Get the model
        either use an index or pass the parameters
        To display the combinations `display_models` method
        To display the specific kwargs of a model see self.imported_snewpy_models[base][1]
        """
        # import and get the model with parameter combinations
        self._import_models(base, sntype)
        par_combinations = self.imported_snewpy_models[base][1]
        model = self.imported_snewpy_models[base][0]
        # if a combination index is passed, select the parameters and return the model
        if combination_index is not None:
            selected_parameters = par_combinations.loc[combination_index].values
            return model(**dict(zip(par_combinations.columns, selected_parameters)))
        else:
            # if parameters are passed, return the model with given parameters
            return model(**kwargs)

    def __call__(self, model_name, combination_index=None, **kwargs):
        """Call snewpy model either with a combination index or with parameters"""
        return self.get_model(model_name, nr_combination=combination_index, **kwargs)
