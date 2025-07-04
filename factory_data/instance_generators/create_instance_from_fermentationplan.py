import random
from src.entities import FactoryData, Instance
import json
import pandas as pd
import numpy as np

### DATA PROCESSING FUNCTIONS
from factory_data.data_reading_functions.read_recipe_data import read_recipes, create_uprod_product_from_recipe
from factory_data.data_reading_functions.read_contamination_combinations import read_contamination_combinations
from factory_data.data_reading_functions.process_tanks import get_modes_v300, get_modes_harvesting, get_modes_UF

### FILEPATHS ###
dir = 'factory_data/seclin_recipes'
contamination_path = f'{dir}/FAM and MF contamination.csv'
def_enzymes_path = f'{dir}/File_D_DEF_Enzymes.csv'
dsp_recipes = f"{dir}/Recipes DSP.xlsx"
ferm_recipes = f"{dir}/Recipes fermentation.xlsx"

### SETTINGS ###
for FLEXIBLE in [False]:

    # Read enzymes
    enzymes_translation = pd.read_csv(def_enzymes_path)
    print(enzymes_translation)

    # TODO: read file paths from one local point

    # Constants
    # TODO: read from file
    constants = {"max_wait_harv": 5,
                 "max_wait_fam": 5,
                 "max_wait_uf": 5,
                 "max_wait_stab": 5,
                 "cleaning_time_contamination": 12}

    ### READ RECIPES FOR FERMENTATION AND DOWNSTREAM PROCESSING ###
    recipes, translation_table = read_recipes(ferm_recipes, dsp_recipes)
    translation_dict = translation_table.set_index('key').to_dict(orient='index')
    recipes = recipes.set_index('key').to_dict(orient='index')

    ### READ CONTAMINATION PAIRS FOR SEQUENCE-DEPENDENT SETUP TIMES ###
    pairs_contamination = read_contamination_combinations(contamination_path, def_enzymes_path, translation_table)

    ### MACHINE MODES BASED ON MACHINE PREFERENCES ###
    # Obtain allowed machine modes harvesting
    allowed_modes_harvesting = get_modes_harvesting(recipes, translation_table)
    print(allowed_modes_harvesting)

    # Obtain allowed machine modes UF
    allowed_modes_UF = get_modes_UF(translation_table)

    # Obtain allowed machine modes V300
    allowed_modes_v300 = get_modes_v300(recipes, translation_table)

    # Make factory object
    factory = FactoryData(name="Seclin",
                          resource_names=["V100", "V140", "V200", "V218", "V42", "V01", "FAM", "MF", "F+L", "UF",
                                          "FAP1", "V300"],
                          capacity=[1, 1, 1, 1, 1, 6, 3, 1, 4, 4, 5, 11],
                          constants=constants,
                          pairs_contamination=pairs_contamination)

    for key in recipes.keys():
        product = create_uprod_product_from_recipe(key=key,
                                                   recipe=recipes[key],
                                                   translation=translation_dict[key],
                                                   harvesting_tanks=allowed_modes_harvesting[key],
                                                   UF_tanks=allowed_modes_UF[key],
                                                   v300_tanks=allowed_modes_v300[key],
                                                   flexible_fermenters=FLEXIBLE)

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

    # TODO: read fermentation plan and connect to recipe keys
    name_plan = "CIN"
    ferm_plan = pd.read_csv(f"factory_data/fermentation_plans/{name_plan}.csv", delimiter=";")
    ferm_plan['MF'] = ferm_plan['MF (%)'].replace(100, 1)
    ferm_plan['Stab'] = ferm_plan['MF (%)'].replace(100, 1)

    print(f'length plan before merge {len(ferm_plan)}')
    merged_plan = pd.merge(ferm_plan, translation_table, how='left', on=['SKU EoF', 'Fermenter', 'MF', 'Stab'])
    print(f'length plan after merge {len(merged_plan)}')

    merged_plan['key'] = merged_plan['key'].replace(['NaN', 'None', ''], np.nan)
    product_ids = merged_plan['key'].dropna()
    product_ids = product_ids.tolist()

    print(f'We are now creating an instance with {product_ids}')
    print(product_ids)

    nr_products = len(product_ids)
    due_dates = [random.randint(20, 80) for _ in range(nr_products)]  # random due dates
    instance = Instance(product_ids, due_dates, objective_weights, factory)

    # Save to json file
    json_string = json.dumps(instance.to_dict(), indent=4)

    # Specify the file name
    file_name = f"factory_data/uprod_instances/instance_{name_plan}_FLEX={FLEXIBLE}.json"

    # Write the JSON string to the file
    with open(file_name, "w") as file:
        file.write(json_string)

    print(f"Data successfully written to {file_name}")