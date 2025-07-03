import pandas as pd
import numpy as np
from src.entities import ProductData, TaskData, TemporalRelation, ConstraintType


def read_recipes(fermentation_recipes_path: str, dsp_recipes_path: str):
    """
    Preprocesses the fermentation and downstream processing recipes which is needed to create
    the UPROD instances. It returns the merged and preprocessed data as a pandas DataFrame.

    Args:
        fermentation_recipes_path (str): The path to the fermentation recipes.
        dsp_recipes_path (str): The path to the downstream processing recipes.

    Returns:
        merged_df (DataFrame): The merged and preprocessed recipes.

    """

    # Read input data
    ferm_recipes = pd.read_excel(fermentation_recipes_path)
    dsp_recipes = pd.read_excel(dsp_recipes_path)

    # Ferm recipes filtering
    ferm_recipes["V01 tanks (#)"] = (np.minimum(ferm_recipes["V01 group1 (kg)"], 1) +
                                     np.minimum(ferm_recipes["V01 group2 (kg)"], 1) +
                                     np.minimum(ferm_recipes["V01 group3 (kg)"], 1) +
                                     np.minimum(ferm_recipes["V01 group4 (kg)"], 1))

    # Create new columns for processing times
    ferm_recipes["fermentation_prep"] = ferm_recipes['Fermentation__prep (hrs)'] + ferm_recipes['Maintenance__Before prep (hrs)']
    ferm_recipes["fermentation_post"] = ferm_recipes['Fermentation__post (hrs)'] + ferm_recipes['Maintenance__After post (hrs)']
    ferm_recipes["fermentation_time"] = ferm_recipes["Fermentation__process (hrs)"]

    #dsp_recipes = dsp_recipes[dsp_recipes['Stab__In V300'] != 'Yes']  # For now only consider without stab
    # TODO: we can probably base this on the table with Intermediate 1
    dsp_recipes['MF'] = dsp_recipes['Name'].str.contains('MF', na=False)
    dsp_recipes['STAB'] = dsp_recipes['Stab__In V300'].str.contains('Yes', na=False)
    # dsp_recipes = dsp_recipes[dsp_recipes['Stab__In V300'] != 'Yes']  # For now only consider without stab
    dsp_recipes["harvesting_time"] = dsp_recipes["Broth killing (hrs)"] + dsp_recipes["Harvest tanks__Broth preparation (hrs)"]

    dsp_recipes["UF_fractions"] = dsp_recipes['UF__UF fractions (#)']
    dsp_recipes["FAM/MF_time"] = (dsp_recipes['Harvest tanks__End weight (kg)'] / dsp_recipes['UF__UF fractions (#)']) / dsp_recipes['FAM or MF__Process (kg/hr)']
    dsp_recipes["UF_time"] = (dsp_recipes['FAM or MF__Weight (kg at F+L)'] / dsp_recipes['UF__UF fractions (#)']) / dsp_recipes["UF__Process (kg/hr)"]

    dsp_recipes["stab_time"] = dsp_recipes['Stab__Process (hrs)']

    # Merge on 'SKU EoF' and 'Fermenter'
    merged_df = dsp_recipes.merge(ferm_recipes, on=['SKU EoF', 'Fermenter'], how='left', suffixes=('y', ''))

    # Drop duplicate columns created by the merge
    merged_df = merged_df[['SKU Interm1', 'SKU EoF', 'Fermenter', 'Batch weight (kg)', 'fermentation_prep', 'fermentation_time',
                           'fermentation_post', 'harvesting_time', 'FAM/MF_time', 'UF_time', 'stab_time',
                           "UF_fractions", 'UF__Weight ccUF (kg)', 'MF', 'STAB']]

    # Save or display the merged DataFrame
    #merged_df = merged_df.dropna()
    #merged_df['sku_ferm'] = merged_df["SKU EoF"].astype(str) + '_' + merged_df['SKU Interm1'].astype(str) + '_' + merged_df[
       # 'Fermenter'].astype(str)  # Concatenating columns A and B
    merged_df['sku_ferm'] = (merged_df["SKU EoF"].astype(str) + '_' + merged_df['Fermenter'].astype(str) + '_'
                             + merged_df['MF'].astype(str) + '_' + merged_df['STAB'].astype(str))  # Concatenating columns A and B

    # Drop duplicates based on sku_ferm, keeping the first occurrence
    #merged_df = merged_df.drop_duplicates(subset='sku_ferm', keep='first')

    # TODO: find out how important these duplicates actually are

    return merged_df


def create_uprod_product_from_recipe(id: int, name: str, recipe: dict,
                                     harvesting_tanks: list[int], v300_tanks: list[int], flexible_fermenters: bool = True):
    """
    Creates a ProductData object from a preprocessed recipe data given a product ID and

    Args:
        id (int): The product id.
        name (str): The product name.
        recipe (dict): The recipe data needed to fill the ProductData.
        harvesting_tanks (list[tuple[int]]): Tuples of the id's of the harvesting tanks that can be used for harvesting
                                             step of this enzyme recipe.
        v300_tanks (list[tuple[int]]): Tuples of the id's of the v300 tanks that can be used for the stabilization step
                                       of this enzyme recipe.
        flexible_fermenters (bool): Whether all fermenters can be used to do the fermentation step

    Returns:
          product (ProductData): A ProductData object.
    """

    # TODO: maybe it is better to have this as a class method ?
    # TODO: read this from an input table
    print(f'Recipe {name} has {recipe}')
    if recipe["MF"] == True:
        MF = True
    else:
        MF = False

    MAX_WAIT_FAM_MF = 1
    MAX_WAIT_STAB = 1
    MAX_WAIT_HARVEST = 1
    MAX_WAIT_UF = 1

    # The uprod production consist of the following tasks that must be scheduled
    task_names = ["fermentation",
                  "fermentation_prep",
                  "fermenter_during_harvesting",
                  "fermentation_post",
                  "STAB",
                  "harvesting",
                  "V01_buffer"
                  ]

    # Get the processing times from the recipe
    processing_times = {"fermentation_prep": recipe["fermentation_prep"],
                        "fermentation_post": recipe["fermentation_post"],
                         "fermentation": recipe["fermentation_time"],
                         "harvesting": recipe["harvesting_time"],
                         "FAM/MF": round(recipe["FAM/MF_time"]),
                         "UF": round(recipe["UF_time"]),
                         "STAB": recipe["stab_time"]}

    # Get the resource modes for the harvesting tasks
    harvesting_translation = {1: "V01_1", 2: "V01_2", 3: "V01_3", 4: "V01_4", 5: "V01_5", 6: "V01_6"}
    harvesting_modes = []
    for j in harvesting_tanks:
        harvesting_modes.append([harvesting_translation[i] for i in j])

    # Get the resource modes for the V300 tasks
    v300_translation = {1: "V300_1", 2: "V300_2", 3: "V300_3", 4: "V300_4", 5: "V300_5", 6: "V300_6",
                        7: "V300_7", 8: "V300_8", 9: "V300_9", 10: "V300_10", 11: "V300_11"
                        }

    v300_modes = []
    for j in v300_tanks:
        v300_modes.append([v300_translation[i] for i in j])

    # Get the resource modes for the fermentation tasks
    if flexible_fermenters:
        fermenter_modes = ["V100", "V140", "V200", "V218", "V42"]
    else:
        print(f'fermenters are not flexible ')
        # TODO: do this in a more descent way
        fermenter_modes = [name.split('_')[1]]


    # TODO: get this from machine files?
    machine_modes = {"fermentation_prep": fermenter_modes,
                    "fermentation_post": fermenter_modes,
                    "fermentation": fermenter_modes,
                    "fermenter_during_harvesting": fermenter_modes,
                    "STAB": v300_modes,
                    "F+L": ["F+L_1", "F+L_2", "F+L_3", "F+L_4"],
                     "harvesting": harvesting_modes,
                     "V01_buffer": harvesting_modes
                     }

    # Get the task duration from the recipe data
    task_durations = {"fermentation_prep": processing_times["fermentation_prep"],
                          "fermentation_post": processing_times["fermentation_post"],
                          "fermentation": processing_times["fermentation"],
                          "harvesting": processing_times["harvesting"],
                          "fermenter_during_harvesting": processing_times["harvesting"],
                          "FAM/MF": processing_times["FAM/MF"],
                          "V01_buffer": processing_times["FAM/MF"],
                          "F+L": processing_times["FAM/MF"],
                          "STAB": processing_times["STAB"]}

    # These constants are needed to model additional constraints
    constants = {"UF_fractions": recipe["UF_fractions"], "FAM/MF_duration": round(recipe["FAM/MF_time"]),
                            "UF_duration": round(recipe["UF_time"]),}

    # Create product
    product = ProductData(id=id, name=name, constants=constants)

    id = 0
    for task_name in task_names:
        if task_name == "V01_buffer":
            fixed_duration = False
        else:
            fixed_duration = True
        task = TaskData(id=id,
                        name=task_name,
                        duration=task_durations[task_name],
                        resource_modes=machine_modes[task_name],
                        fixed_duration=fixed_duration
                        )
        product.add_task(task)
        id += 1

    # First, handle the constraints for fermentation and harvesting
    start_before_start_constraints = []

    start_before_end_constraints = [("fermentation", "fermentation_prep", None),
                                    ("fermentation_post", "fermenter_during_harvesting", None),
                                    ("harvesting", "fermentation", - MAX_WAIT_HARVEST)]

    end_before_start_constraints = [("fermentation_prep", "fermentation", None),
                                    ("fermenter_during_harvesting", "fermentation_post", None),
                                    ("fermentation", "harvesting", None)]

    end_before_end_constraints = []

    identical_resources = [("fermentation_prep", "fermentation"),
                           ("fermentation_post", "fermentation"),
                           ("fermenter_during_harvesting", "fermentation")]

    # Handle the fractions for downstream processing
    nr_fracs = constants["UF_fractions"]

    if MF:
        FAM_MF_modes = ["MF"]
    else:
        FAM_MF_modes = ["FAM_1", "FAM_2", "FAM_3"]

    # For each UF fraction we make a FAM/MF, FAP1, UF and F+L task that require machines
    for uf_frac in range(nr_fracs):
        # TODO: make consistency in task_durations and processing_times dict
        dsp_tasks = [(f"FAM/MF_frac_{uf_frac}", task_durations["FAM/MF"], FAM_MF_modes, True),
                     (f"FAP1_frac_{uf_frac}", processing_times["UF"], ["FAP1_1", "FAP1_2", "FAP1_3", "FAP1_4", "FAP1_5"], True),
                     (f"UF_frac_{uf_frac}", processing_times["UF"], ["UF_1", "UF_2", "UF_3", "UF_4"], True),
                     (f"F+L_frac_{uf_frac}",  task_durations["FAM/MF"], ["F+L_1", "F+L_2", "F+L_3", "F+L_4"], False)]

        # Create task entities and add to product
        for (name, duration, resource_modes, fixed_duration) in dsp_tasks:

            task = TaskData(id=id,
                            name=name,
                            duration=duration,
                            resource_modes=resource_modes,
                            fixed_duration=fixed_duration
                            )
            product.add_task(task)
            id += 1

        start_before_start_extra = [(f"F+L_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", None),
                                    (f"FAM/MF_frac_{uf_frac}", f"F+L_frac_{uf_frac}", None),
                                    (f"FAP1_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                    (f"UF_frac_{uf_frac}", f"FAP1_frac_{uf_frac}", None),
                                    (f"FAM/MF_frac_{uf_frac}", f"UF_frac_{uf_frac}", int(0.9 * task_durations["FAM/MF"]))
                                    ]

        start_before_end_extra = [(f"UF_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", -MAX_WAIT_UF)]

        end_before_start_extra = [("harvesting", f"FAM/MF_frac_{uf_frac}", None)
                                  ]

        end_before_end_extra = [(f"FAP1_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                (f"UF_frac_{uf_frac}", f"FAP1_frac_{uf_frac}", None),
                                (f"F+L_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                (f"UF_frac_{uf_frac}", f"F+L_frac_{uf_frac}", None)]

        start_before_start_constraints += start_before_start_extra

        start_before_end_constraints += start_before_end_extra

        end_before_start_constraints += end_before_start_extra

        end_before_end_constraints += end_before_end_extra

    # Buffer tanks during the FAM/MF/ MF
    # Harvesting tank is still occupied during the FAM/MF operations
    end_before_end_constraints += [(f"FAM/MF_frac_{nr_fracs - 1}", "V01_buffer", None)]

    # Fermenter tank in use during fermentation
    start_before_start_constraints += [("V01_buffer", f"FAM/MF_frac_0", None),
                           ("harvesting", "fermenter_during_harvesting", None),
                           ("fermenter_during_harvesting", "harvesting", None)]

    start_before_end_constraints += [("STAB", f"UF_frac_{uf_frac}", - MAX_WAIT_STAB),
                                     (f"FAM/MF_frac_0", "harvesting", - MAX_WAIT_FAM_MF)]

    # V300 after last UF fraction
    end_before_start_constraints += [(f"UF_frac_{uf_frac}", "STAB", None)]

    # Add consecutive machines
    identical_resources += [("V01_buffer", "harvesting")]

    # Temporal constraints that only apply to the final one or first one
    # Ordering of the fractions
    for frac in range(nr_fracs - 1):
        start_before_start_constraints += [(f"UF_frac_{frac}", f"UF_frac_{frac + 1}", None)]

    # Convert all collected constraints into objects and add to product object
    for (task1, task2, delay) in start_before_start_constraints:
        temp_cons = TemporalRelation(task1, task2, delay, ConstraintType.StartToStart)
        product.add_temporal_constraints(temp_cons)

    for (task1, task2, delay) in start_before_end_constraints:
        temp_cons = TemporalRelation(task1, task2, delay, ConstraintType.StartToFinish)
        product.add_temporal_constraints(temp_cons)

    for (task1, task2, delay) in end_before_start_constraints:
        temp_cons = TemporalRelation(task1, task2, delay, ConstraintType.FinishToStart)
        product.add_temporal_constraints(temp_cons)

    for (task1, task2, delay) in end_before_end_constraints:
        temp_cons = TemporalRelation(task1, task2, delay, ConstraintType.FinishToFinish)
        product.add_temporal_constraints(temp_cons)

    # Add consecutive machines during fermentation and harvesting
    product.identical_resources = identical_resources

    return product


if __name__ == '__main__':
    # Read input data
    dsp_recipes = "factory_data/seclin_recipes/Recipes DSP.xlsx"
    ferm_recipes = "factory_data/seclin_recipes/Recipes fermentation.xlsx"

    recipes = read_recipes(ferm_recipes, dsp_recipes)
