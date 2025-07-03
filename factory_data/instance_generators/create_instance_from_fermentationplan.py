import random
from src.entities import FactoryData, Instance
import json
import pandas as pd

### FILEPATHS ###
dir = 'factory_data/seclin_recipes'
contamination_path = f'{dir}/FAM and MF contamination.csv'
def_enzymes_path = f'{dir}/File_D_DEF_Enzymes.csv'
dsp_recipes = f"{dir}/Recipes DSP.xlsx"
ferm_recipes = f"{dir}/Recipes fermentation.xlsx"

### SETTINGS ###
for FLEXIBLE in [False]:

    # Construct factory object for seclin
    from factory_data.data_reading_functions.read_contamination_combinations import read_contamination_combinations
    from factory_data.data_reading_functions.read_recipe_data import read_recipes, create_uprod_product_from_recipe
    from factory_data.data_reading_functions.process_tanks import get_modes_v300, get_modes_harvesting

    # Read enzymes
    enzymes_translation = pd.read_csv(def_enzymes_path)
    print(enzymes_translation)

    # TODO: read file paths from one local point
    pairs_contamination = read_contamination_combinations(contamination_path, def_enzymes_path)

    # Constants
    # TODO: read from file
    constants = {"max_wait_harv": 1,
                 "max_wait_fam": 1,
                 "max_wait_uf": 1,
                 "max_wait_stab": 1,
                 "fam_cleaning_contamination": 3}

    # Make factory object
    factory = FactoryData(name="Seclin",
                      resource_names=["V100", "V140", "V200", "V218", "V42", "V01", "FAM", "MF", "F+L", "UF", "FAP1", "V300"],
                      capacity=[1, 1, 1, 1, 1, 6, 3, 1, 4, 4, 5, 11],
                      constants=constants,
                      pairs_contamination=pairs_contamination)

    # Read recipes and add to factory data
    recipes = read_recipes(ferm_recipes, dsp_recipes)
    recipes = recipes.set_index('sku_ferm').to_dict(orient='index')

    # Create products for instance
    product_keys = [key for key in recipes.keys()]
    print(product_keys)

    # Obtain allowed machine modes harvesting
    allowed_modes_harvesting = get_modes_harvesting(recipes)

    # Obtain allowed machine modes V300
    allowed_modes_v300 = get_modes_v300(recipes)

    id = 0

    for key in product_keys:
        product = create_uprod_product_from_recipe(id=id,
                                                   name=key,
                                                   recipe=recipes[key],
                                                   harvesting_tanks=allowed_modes_harvesting[key],
                                                   v300_tanks=allowed_modes_v300[key],
                                                   flexible_fermenters=FLEXIBLE)

        factory.add_product(product)
        id += 1


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

    # TODO: read fermentation plan
    name_plan = "plan_july_2024"
    fermentation_plan = pd.read_csv(f"factory_data/fermentation_plans/{name_plan}.csv", delimiter=";")
    fermentation_plan['MF'] = fermentation_plan['MF (%)'] == 100
    fermentation_plan['STAB'] = fermentation_plan['Stab (%)'] == 100

    # TODO: make new column with SKU - fermenter combination
    fermentation_plan['key'] = (fermentation_plan["SKU EoF"].astype(str) + '_' + fermentation_plan['Fermenter'].astype(str) + '_'
                                 + fermentation_plan['MF'].astype(str) + '_' + fermentation_plan['STAB'].astype(str))  # Concatenating columns A and B
    selected_keys = fermentation_plan["key"].tolist()

    print(f'We are now creating an instance with {selected_keys}')

    product_ids = []

    # TODO: this is now a repair mechanism that we just use because
    for name in selected_keys:
        if name in product_keys:
            use_name = name
        else:
            print(f'Name {name} not in keys')
            # Split by underscore and replace first 'True' with 'False'
            parts = name.split('_')
            for i, part in enumerate(parts):
                if part == 'True':
                    parts[i] = 'False'
                    break  # Only change the first occurrence
            fallback_name = '_'.join(parts)

            # Try fallback
            if fallback_name in product_keys:
                use_name = fallback_name
            else:
                use_name = None
        print(f'Using name {use_name}')
        product_ids.append(product_keys.index(use_name))
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