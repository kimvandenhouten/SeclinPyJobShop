import random
from src.entities import FactoryData, Instance
import json
import pandas as pd
import numpy as np
import os

### DATA PROCESSING FUNCTIONS
from factory_data.data_reading_functions.read_data_files import load_dataframes_from_files
from factory_data.data_reading_functions.read_recipe_data import create_recipe_dict, create_product_from_recipe
from factory_data.data_reading_functions.read_contamination_combinations import read_contamination_combinations
from factory_data.data_reading_functions.process_tanks import get_modes_v300, get_modes_harvesting, get_modes_UF, read_equipment_seclin

# Old data
dir_path = "factory_data/seclin_recipes"
filenames = ["DEF Enzymes.csv", "F and L tanks.csv", "FAM and MF.csv", "FAM and MF contamination.csv", "FAM and MF preferences.csv",
             "FAP.csv", "Intermediate 1.csv", "UF preferences.csv", "Harvest tanks E.csv", "Harvest tanks V01.csv",
             "Harvesting preferences V01.csv", "V300 preferences.csv", "V300 tanks.csv", "Recipes DSP.xlsx", "Recipes fermentation.xlsx"]
old_data = load_dataframes_from_files(dir_path, filenames)

dir_path = "factory_data/csv_inputs"
filenames = os.listdir(dir_path)
print(f'file names are {filenames}')
data = load_dataframes_from_files(dir_path, filenames)

# TODO: ask Eva, read harvesting preferences
data["Harvesting preferences V01"] = old_data["Harvesting preferences V01"]
data["DEF Enzymes"] = old_data["DEF Enzymes"]

# READ FERMENTATION PLAN
ferm_plan = pd.read_csv("factory_data/csv_inputs/Fermentation Plan.csv")

# Constants
# TODO: read from file
constants = {"max_wait_harv": 5,
             "max_wait_fam": 5,
             "max_wait_uf": 5,
             "max_wait_stab": 5,
             "cleaning_time_contamination": 12}

### READ RECIPES FOR FERMENTATION AND DOWNSTREAM PROCESSING ###
recipes, translation_table = create_recipe_dict(data)
recipes = recipes[recipes["FAM/MF_time"] != np.inf]
recipes.to_csv("merged_recipes.csv")
print(recipes)
translation_dict = translation_table.set_index('key').to_dict(orient='index')
recipe_dict = recipes.set_index('key').to_dict(orient='index')

### READ CONTAMINATION PAIRS FOR SEQUENCE-DEPENDENT SETUP TIMES ###
contamination_path = f'factory_data/seclin_recipes/FAM and MF contamination.csv'
def_enzymes_path = f'factory_data/seclin_recipes/DEF Enzymes.csv'
pairs_contamination = read_contamination_combinations(contamination_path, def_enzymes_path, translation_table)

### MACHINE MODES BASED ON MACHINE PREFERENCES ###
# Read all resources from Seclin
resource_names = read_equipment_seclin(data)

# Obtain all Etanks
etanks = data["Harvest tanks E"]["Tank"].tolist()

# Obtain all F+L tanks
fl_tanks = data["F and L tanks"]["Equipment"].tolist()

# Obtain allowed machine modes harvesting
allowed_modes_harvesting = get_modes_harvesting(translation_table, data, recipe_dict)

# Obtain allowed machine modes UF
allowed_modes_UF = get_modes_UF(translation_table, data)
print(f'finish getting modes UF')

# Obtain allowed machine modes V300
allowed_modes_v300 = get_modes_v300(translation_table, data, recipe_dict)
print(f'finish getting modes v300')

# Make factory object
factory = FactoryData(name="Seclin",
                      resource_names=resource_names,
                      capacity=[1 for i in resource_names],
                      constants=constants,
                      pairs_contamination=pairs_contamination)


for key in recipe_dict.keys():
    print(f'We create a product with key {key}, that product requires {recipe_dict[key]["Etanks required (#)"]}')
    product = create_product_from_recipe(key=key,
                                         recipe=recipe_dict[key],
                                         translation=translation_dict[key],
                                         etanks=etanks,
                                         fandltanks=fl_tanks,
                                         harvesting_tanks=allowed_modes_harvesting[key],
                                         UF_tanks=allowed_modes_UF[key],
                                         v300_tanks=allowed_modes_v300[key],
                                         flexible_fermenters=False)

    factory.add_product(product)


### Construct an instance
# Set objective weights
objective_weights = {
    "weight_makespan": 1,
    "weight_tardy_jobs": 0,
    "weight_total_flow_time": 0,
    "weight_total_tardiness": 0,
    "weight_total_earliness": 0,
    "weight_max_tardiness": 0,
    "weight_max_lateness": 0,
}

# Connect fermentation plan to recipes
print(ferm_plan)
ferm_plan['MF'] = ferm_plan['MF (%)'].replace(100, 1)
ferm_plan['Stab'] = ferm_plan['MF (%)'].replace(100, 1)

merged_plan = pd.merge(ferm_plan, translation_table, how='left', on=['SKU EoF', 'Fermenter', 'MF', 'Stab'])

merged_plan['key'] = merged_plan['key'].replace(['NaN', 'None', ''], np.nan)
product_ids = merged_plan['key'].dropna()
product_ids = product_ids.tolist()

nr_products = len(product_ids)
due_dates = [random.randint(20, 80) for _ in range(nr_products)]  # random due dates
instance = Instance(product_ids, due_dates, objective_weights, factory)

# Save to json file
json_string = json.dumps(instance.to_dict(), indent=4)

# Specify the file name
file_name = f"factory_data/uprod_instances/instance_plan_july.json"

# Write the JSON string to the file
with open(file_name, "w") as file:
    file.write(json_string)

print(f"Data successfully written to {file_name}")