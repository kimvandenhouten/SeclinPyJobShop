import random
from src.entities import FactoryData, Instance, TemporalRelation, ConstraintType, CPModel
import json

### FILEPATHS ###
dir = 'factory_data/seclin_recipes'
contamination_path = f'{dir}/FAM and MF contamination.csv'
def_enzymes_path = f'{dir}/File_D_DEF_Enzymes.csv'
dsp_recipes = f"{dir}/Recipes DSP.xlsx"
ferm_recipes = f"{dir}/Recipes fermentation.xlsx"

# Construct factory object for seclin
from factory_data.data_reading_functions.read_contamination_combinations import read_contamination_combinations
from factory_data.data_reading_functions.read_recipe_data import read_recipes, create_uprod_product_from_recipe
from factory_data.data_reading_functions.process_tanks import get_modes_v300, get_modes_harvesting

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
                  resource_names=["Fermenter", "V01", "FAM", "MF", "F+L", "UF", "FAP1", "V300"],
                  capacity=[5, 6, 3, 1, 4, 4, 5, 11],
                  constants=constants,
                  pairs_contamination=pairs_contamination)

# Read recipes and add to factory data
recipes = read_recipes(ferm_recipes, dsp_recipes)
recipes = recipes.set_index('sku_ferm').to_dict(orient='index')

# Create products for instance
product_keys = [key for key in recipes.keys()]

# Obtain allowed machine modes harvesting
allowed_modes_harvesting = get_modes_harvesting(recipes)

# Obtain allowed machine modes V300
allowed_modes_v300 = get_modes_v300(recipes)

id = 0
selected_key = '20322_23322_V140'
print(product_keys)
for key in product_keys:
    if key == selected_key:
        print(f'Allowed modes harvesting {allowed_modes_harvesting[key]}')
        print(f'id is {id}')
    product = create_uprod_product_from_recipe(id=id,
                                               name=key,
                                               recipe=recipes[key],
                                               harvesting_tanks=allowed_modes_harvesting[key],
                                               v300_tanks=allowed_modes_v300[key])

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

# Set instance data
for key in [selected_key]:

    product_ids = [product_keys.index(key)]
    nr_products = len(product_ids)
    due_dates = [random.randint(20, 80) for _ in range(nr_products)]  # random due dates
    instance = Instance(product_ids, due_dates, objective_weights, factory)

    # Save to json file
    json_string = json.dumps(instance.to_dict(), indent=4)

    # Specify the file name
    file_name = f"factory_data/uprod_instances/instance_{selected_key}.json"

    # Write the JSON string to the file
    with open(file_name, "w") as file:
        file.write(json_string)

    print(f"Data successfully written to {file_name}")