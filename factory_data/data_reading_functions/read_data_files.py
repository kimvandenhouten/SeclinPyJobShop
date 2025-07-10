import pandas as pd
### FILEPATHS ###
dir = 'factory_data/seclin_recipes'
contamination_path = f'{dir}/FAM and MF contamination.csv'
def_enzymes_path = f'{dir}/File_D_DEF_Enzymes.csv'
dsp_recipes_path = f"{dir}/Recipes DSP.xlsx"
ferm_recipes_path = f"{dir}/Recipes fermentation.xlsx"
ferm_recipes_path = "factory_data/InputData/Recipes fermentation.xlsx"
harvest_tanks_path = f"{dir}/file_D_Harvest tanks V01.csv"
name_plan = "parallel_FAM"
ferm_plan = pd.read_csv(f"factory_data/fermentation_plans/{name_plan}.csv", delimiter=";")

import os
import pandas as pd


def load_dataframes_from_files(dir_path, filenames):
    """
    Load .csv and .xlsx files from a directory into a dictionary of DataFrames.

    Args:
        dir_path (str): The directory containing the files.
        filenames (list): List of filenames (must end with .csv or .xlsx).

    Returns:
        dict: A dictionary where keys are filenames without extensions,
              and values are pandas DataFrames.
    """
    dataframes = {}
    for filename in filenames:
        full_path = os.path.join(dir_path, filename)
        name, ext = os.path.splitext(filename)

        if ext == '.csv':
            df = pd.read_csv(full_path)
        elif ext == '.xlsx':
            df = pd.read_excel(full_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        dataframes[name] = df

    return dataframes