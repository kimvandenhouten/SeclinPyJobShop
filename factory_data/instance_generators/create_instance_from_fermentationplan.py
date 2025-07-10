import random
from src.entities import FactoryData, Instance
import json
import pandas as pd
import numpy as np

### DATA PROCESSING FUNCTIONS
from factory_data.data_reading_functions.read_data_files import load_dataframes_from_files
from factory_data.data_reading_functions.read_recipe_data import create_recipe_dict, create_product_from_recipe
from factory_data.data_reading_functions.read_contamination_combinations import read_contamination_combinations
from factory_data.data_reading_functions.process_tanks import get_modes_v300, get_modes_harvesting, get_modes_UF

dir_path = "factory_data/seclin_recipes"
filenames = ["DEF Enzymes.csv", "F and L tanks.csv", "FAM and MF.csv", "FAM and MF contamination.csv", "FAM and MF preferences.csv",
             "FAP.csv", "Intermediate 1.csv", "UF preferences.csv", "Harvest tanks E.csv", "Harvest tanks V01.csv",
             "Harvesting preferences V01.csv", "V300 preferences.csv", "V300 tanks.csv", "Recipes DSP.xlsx", "Recipes fermentation.xlsx"]
data = load_dataframes_from_files(dir_path, filenames)
print(data.keys())
# READ FERMENTATION PLAN
ferm_plan = pd.read_csv("factory_data/fermentation_plans/parallel_FAM.csv")

# Constants
# TODO: read from file
constants = {"max_wait_harv": 5,
             "max_wait_fam": 5,
             "max_wait_uf": 5,
             "max_wait_stab": 5,
             "cleaning_time_contamination": 12}

### READ RECIPES FOR FERMENTATION AND DOWNSTREAM PROCESSING ###
recipes, translation_table = create_recipe_dict(data)
print(recipes)
translation_dict = translation_table.set_index('key').to_dict(orient='index')
recipe_dict = recipes.set_index('key').to_dict(orient='index')

### READ CONTAMINATION PAIRS FOR SEQUENCE-DEPENDENT SETUP TIMES ###
contamination_path = f'{dir_path}/FAM and MF contamination.csv'
def_enzymes_path = f'{dir_path}/DEF Enzymes.csv'
pairs_contamination = read_contamination_combinations(contamination_path, def_enzymes_path, translation_table)

### MACHINE MODES BASED ON MACHINE PREFERENCES ###
# Obtain allowed machine modes harvesting
allowed_modes_harvesting = get_modes_harvesting(translation_table, data, recipe_dict)

# Obtain allowed machine modes UF
allowed_modes_UF = get_modes_UF(translation_table, data)

# Obtain allowed machine modes V300
allowed_modes_v300 = get_modes_v300(translation_table, data, recipe_dict)

# Make factory object
factory = FactoryData(name="Seclin",
                      resource_names=["V100", "V140", "V200", "V218", "V42", "E", "V01", "FAM", "MF", "F+L", "UF", "FAP1", "V300"],
                      capacity=[1, 1, 1, 1, 1, 4, 6, 3, 1, 4, 4, 5, 11],
                      constants=constants,
                      pairs_contamination=pairs_contamination)

for key in recipe_dict.keys():
    print(f'We create a product with key {key}, that product requires {recipe_dict[key]["Etanks required (#)"]}')
    product = create_product_from_recipe(key=key,
                                         recipe=recipe_dict[key],
                                         translation=translation_dict[key],
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
file_name = f"factory_data/uprod_instances/instance.json"

# Write the JSON string to the file
with open(file_name, "w") as file:
    file.write(json_string)

print(f"Data successfully written to {file_name}")