
import os, click
from glob import glob
import re
from inspect import signature
import snewpy
import numpy as np
from snewpy.models.ccsn import Bollig_2016, Fornax_2019, Fornax_2021, Kuroda_2020
from snewpy.models.ccsn import Nakazato_2013, OConnor_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014
from snewpy.models.ccsn import Walk_2018, Walk_2019, Warren_2020, Zha_2021

models_list = ['Bollig_2016', 'Fornax_2019', 'Fornax_2021', 'Kuroda_2020', 'Nakazato_2013',
               'OConnor_2013', 'OConnor_2015', 'Sukhbold_2015', 'Tamborra_2014', 'Walk_2018', 'Walk_2019',
               'Warren_2020', 'Zha_2021',]

models = [Bollig_2016, Fornax_2019, Fornax_2021, Kuroda_2020,
          Nakazato_2013, OConnor_2013, OConnor_2015, Sukhbold_2015, Tamborra_2014,
          Walk_2018, Walk_2019, Warren_2020, Zha_2021]

models_dict = dict(zip(models_list, models))


def _get_files_in_model(model_name, config):
    try:
        snewpy_base = config['paths']['snewpy_models']
        file_path = os.path.join(snewpy_base, model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0
        return files_in_model, file_path
    except Exception as e:
        print(f"{e} looking at snewpy installation")
        snewpy_base = snewpy.__file__.split("python/snewpy/__init__.py")[0]
        file_path = os.path.join(snewpy_base, "models", model_name)
        files_in_model = glob(os.path.join(file_path, '*'))
        assert len(files_in_model) > 0
        return files_in_model, file_path


def find_warren2020(files_in_model, file_path, index):
    click.secho(f"Warren 2022 is a bit different", fg='blue')
    _files_in_model_list = [f"[{i}]\t" + f.split("/")[-1] for i, f in enumerate(files_in_model)]
    _files_to_print = "\n".join(_files_in_model_list) + "\n"
    if index is None:
        click.secho(f"First, Select an alpha param:", fg='blue')
        file_index = input(_files_to_print)
        selected_folder = files_in_model[int(file_index)]
        files_in_folder = glob(os.path.join(selected_folder, "*"))

        _files_in_folder = [f"[{i}]\t" + f.split("/")[-1].split("stir_multimessenger_")[1].split('.h5')[0] for i, f in
                            enumerate(files_in_folder)]
        _files_to_print = "\n".join(_files_in_folder) + "\n"
        click.secho(f"Now, Select a mass param:", fg='blue')
        file_index = input(_files_to_print)
        selected_file = files_in_folder[int(file_index)]
    else:
        if np.ndim(index) == 0:
            return find_warren2020(files_in_model, file_path, None)
        else:
            folder_index = index[0]
            selected_folder = files_in_model[int(folder_index)]
            files_in_folder = glob(os.path.join(selected_folder, "*"))
            file_index = index[1]
            selected_file = files_in_folder[int(file_index)]
    return selected_file

def _select_shorter_name(model_name, selected_file):
    """ Test the selected path with shorter filenames
        Some models have file names e.g. s27.0_LSS220_nue
        but their respective class only accepts s27.0
        and some only wants the path of the directory and not the name
    """
    filename = selected_file.split('/')[-1]
    found = False
    for i in range(len(filename)+1):
        if i == 0:
            f = selected_file
        else:
            f = selected_file[:-i]
        try:
            models_dict[model_name](f)
            found = True
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            print(f"Some Error Occured {e}\n still trying")
    if found:
        return f
    else:
        raise FileNotFoundError(f"for {model_name} I couldn't find the selected file")


def _parse_models(model_name, filename, index, config):
    """ Get the selected model, or ask user
    """
    possible_kwargs = list(signature(models_dict[model_name]).parameters.keys())[1:]
    files_in_model, file_path = _get_files_in_model(model_name, config)
    files_in_model = [f for f in files_in_model if not (f.endswith('.md') or f.endswith('.ipynb') or f.endswith('.py'))]
    if filename is None:
        if model_name == "Warren_2020":
            return find_warren2020(files_in_model, file_path, index)

        if any([f.endswith("nuebar") for f in files_in_model]):
            # it is structured differently
            files_in_model = list(set([re.split('_nux|_nue|_nuebar', f)[0] for f in files_in_model]))

        _files_in_model_list = [f"[{i}]\t" + f.split("/")[-1] for i, f in enumerate(files_in_model)]
        _files_to_print = "\n".join(_files_in_model_list) + "\n"
        if index is None:
            click.secho(f"> Available files for this {model_name}, please select an index", fg='blue', bold=True)
            click.secho(f"> {model_name}, can take model_kwargs: {possible_kwargs} \n\n", fg='blue', bold=True)
            file_index = input(_files_to_print)
        else:
            file_index = index

        selected_file = files_in_model[int(file_index)]
        click.secho(f"> You chose ~wisely~ ->\t   {_files_in_model_list[int(file_index)]}", fg='blue', bold=True)
        return selected_file
    else:
        if filename in [f.split("/")[-1] for i, f in enumerate(files_in_model)]:
            return os.path.join(file_path, filename)
        else:
            raise FileNotFoundError(f"{filename} not found in {file_path}")


def fetch_model(model_name, filename, index, config, **model_kwargs):
    file = _parse_models(model_name, filename, index, config)
    model = models_dict[model_name](file, **model_kwargs)
    return file, model
