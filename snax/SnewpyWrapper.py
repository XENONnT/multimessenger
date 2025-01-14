import os, sys
import numpy as np
import pandas as pd
import snewpy
from .sn_utils import get_hash_from_model, isnotebook
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

snewpy_models_path = "/project2/lgrandi/xenonnt/simulations/supernova/SNEWPY_models/"
# all params file
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, os.pardir)
allparams_csv_path = os.path.join(parent_dir, 'all_parameters.csv')

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
        self.all_params = None

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
            par_combinations = model.get_param_combinations()
            # get hashes
            model_hashes = []
            for pars in tqdm(par_combinations, total=len(par_combinations)):
                _hash = get_hash_from_model(model(**pars))
                model_hashes.append(_hash)
            par_combinations = pd.DataFrame(par_combinations)
            # par_combinations["model_name"] = np.repeat(model.__name__, len(model_hashes))
            par_combinations["hash"] = model_hashes
            par_combinations["Combination Index"] = np.arange(len(par_combinations)) + 1
            par_combinations.set_index("Combination Index", inplace=True)
            self.imported_snewpy_models[base] = (model, par_combinations)
            return None

    def display_models(self, base=None, sntype="ccsn", verbose=True):
        """Display the available models
        Get the model by calling `get_model('{base}', combination_index=#)`
        or by calling `get_model('{base}', **kwargs)`
        """
        if base is None:
            print(self.model_keys)
        else:
            self._import_models(base, sntype)
            if verbose:
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
            _par_combinations = par_combinations.copy() # copy so that self.imported_snewpy_models is not overwritten
            _par_combinations.drop(columns='hash', inplace=True) # hash is not a par
            selected_parameters = _par_combinations.loc[combination_index].values
            return model(**dict(zip(_par_combinations.columns, selected_parameters)))
        else:
            # if parameters are passed, return the model with given parameters
            return model(**kwargs)

    def _get_all_model_params(self):
        """ Get all the model parameters and hashes
            This function goes through all the models and checks the
            valid parameter combinations, frames everything with their names and hashes
        """
        # Initialize an empty list to store dataframes
        if self.all_params is not None:
            return self.all_params

        try:
            df = pd.read_csv(allparams_csv_path)
            return df
        except Exception as e:
            # continue to check
            print(f"{e} \nGetting all parameters")

        dfs = []
        ccsn_keys = [k for k in self.model_keys if k not in ["PISN", "Type_Ia", "presn-models"]]
        for input_name in ccsn_keys:
            # Call your function to get a dataframe for each input
            df = self.display_models(input_name, verbose=False)
            df['name'] = np.repeat(input_name, len(df))
            # Append the dataframe to the list
            dfs.append(df)

        # Concatenate all dataframes in the list
        result_df = pd.concat(dfs, ignore_index=True)

        # Get unique column names across all dataframes
        unique_columns = set().union(*(set(df.columns) for df in dfs))

        # Fill missing columns with None
        for missing_column in unique_columns:
            if missing_column not in result_df.columns:
                result_df[missing_column] = None

        # rearrange columns
        cols = [c for c in result_df.columns if c not in ["name", "hash"]]
        cols = ["name", "hash"] + cols
        result_df = result_df.loc[:, cols]
        self.all_params = result_df
        return result_df

    def __call__(self, model_name, combination_index=None, **kwargs):
        """Call snewpy model either with a combination index or with parameters"""
        return self.get_model(model_name, nr_combination=combination_index, **kwargs)
