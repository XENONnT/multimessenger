
import os, click
from glob import glob
import re
from inspect import signature
import snewpy
from snewpy.models.ccsn import Bollig_2016, Fornax_2021, Kuroda_2020
from snewpy.models.ccsn import Nakazato_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014
from snewpy.models.ccsn import Walk_2018, Walk_2019, Zha_2021

## for now ignore Farnax_2019, OConnor_2013 and Warren_2020

models_list = ['Bollig_2016', 'Fornax_2021',  'Kuroda_2020', 'Nakazato_2013',
               'OConnor_2015', 'Sukhbold_2015', 'Tamborra_2014',
               'Walk_2018', 'Walk_2019', 'Zha_2021',]

models = [Bollig_2016, Fornax_2021, Kuroda_2020,
          Nakazato_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014,
          Walk_2018, Walk_2019, Zha_2021]

models_dict = dict(zip(models_list, models))


def _get_files_in_folder(model_name, config):
    try:
        snewpy_base = config['paths']['snewpy_models']
        file_path = os.path.join(snewpy_base, model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0,  f"The model folder {file_path} is empty"
        return files_in_model, file_path
    except Exception as e:
        print(f"{e} looking at snewpy installation")
        snewpy_base = snewpy.__file__.split("python/snewpy/__init__.py")[0]
        file_path = os.path.join(snewpy_base, "models", model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0
        return files_in_model, file_path

class SnewpyWrapper:
    """ Load the snewpy models easily
    """
    def __init__(self, name, config):
        self.name = name
        self.config = config
        if name in models_list:
            self.models = models_dict[name]
        else:
            raise KeyError(f"Selected {name} does not exist!")
        self.files_in_model, self.folder_path = _get_files_in_folder(self.name, self.config)
        self.selected_file = None

    def load_model_data(self, filename=None, index=None, force=False, **model_kwargs):
        # TODO: allow for model_kwargs only
        """ Load a specific progenitor data for a given model
        :param filename: `str` filename to load, optional
        :param index: `int` the index of the file to load, optional
        """
        # select a file
        if not force:
            # if not forced, load the already-selected-file
            file_to_load = self.selected_file or self._parse_models(filename, index)
        else:
            # when forced, allow for other file selection
            file_to_load = self._parse_models(filename, index)
        self.selected_file = file_to_load
        model = models_dict[self.name](file_to_load, **model_kwargs)
        return model


    def _select_shorter_name(self, selected_file):
        """ Test the selected path with shorter filenames
            Some models have file names e.g. s27.0_LSS220_nue
            but their respective class only accepts s27.0
            and some only wants the path of the directory and not the name
        """
        filename = selected_file.split('/')[-1]
        found = False
        # keep testing with shorter filename strings
        for i in range(len(filename)+1):
            if i == 0:
                f = selected_file
            else:
                f = selected_file[:-i]
            try:
                models_dict[self.name](f)
                found = True
                break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"Some Error Occured {e}\n still trying")
        if found:
            return f
        else:
            raise FileNotFoundError(f"for {self.name} I couldn't find the selected file")


    def _parse_models(self, filename, index):
        """ Get the selected model, or ask user
        """
        # collect the available data files
        possible_kwargs = list(signature(models_dict[self.name]).parameters.keys())[1:]
        files_in_model = [f for f in self.files_in_model if not (f.endswith('.md') or f.endswith('.ipynb') or f.endswith('.py'))]
        # check if user specified a filename
        if filename is not None:
            # check if it is in the folder for that model
            if filename in [f.split("/")[-1] for i, f in enumerate(files_in_model)]:
                return os.path.join(self.folder_path, filename)
            else:
                raise FileNotFoundError(f"{filename} not found in {self.folder_path}")
        else:
            # if no name is specified, find all files and ask user
            if any([f.endswith("nuebar") for f in files_in_model]):
                # it is structured differently
                files_in_model = list(set([re.split('_nux|_nue|_nuebar', f)[0] for f in files_in_model]))

            _files_in_model_list = [f"[{i}]\t" + f.split("/")[-1] for i, f in enumerate(files_in_model)]
            _files_to_print = "\n".join(_files_in_model_list) + "\n"
            # even if no filename is given, files can also be selected by their index
            if index is None:
                click.secho(f"> Available files for this {self.name}, please select an index", fg='blue', bold=True)
                click.secho(f"> {self.name}, can take model_kwargs: {possible_kwargs} \n\n", fg='blue', bold=True)
                file_index = input(_files_to_print)
            else:
                file_index = index

            # find the selected file
            selected_file = files_in_model[int(file_index)]
            # find the correct "short" name that snewpy accepts
            selected_file = self._select_shorter_name(selected_file)
            click.secho(f"> You chose ~wisely~ ->\t   {_files_in_model_list[int(file_index)]}", fg='blue', bold=True)
            # self.selected_file = selected_file
            return selected_file

