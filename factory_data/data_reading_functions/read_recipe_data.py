import pandas as pd
import numpy as np
from src.entities import ProductData, TaskData, TemporalRelation, ConstraintType
from itertools import combinations


def get_lag(row):
    try:
        if np.isnan(row['min_number_fractions']) or np.isinf(row['min_number_fractions']) or row['min_number_fractions'] == 0:
            return False
        col_name = f"t{int(row['min_number_fractions'])} (hrs)"
        return row[col_name] if col_name in row else False
    except:
        return False


def create_recipe_dict(data):
    """
    Preprocesses the fermentation and downstream processing recipes which is needed to create
    the UPROD instances. It returns the merged and preprocessed data as a pandas DataFrame.

    Args:
        fermentation_recipes_path (Dataframe): The df with the fermentation recipes.
        dsp_recipes_path (Dataframe): The df with the downstream processing recipes.
        harvesting tanks: The df with the harvest tanks.

    Returns:
        merged_df (DataFrame): The merged and preprocessed recipes.

    """
    ferm_recipes = data["Recipes fermentation"]
    dsp_recipes = data["Recipes DSP"]
    harvest_tanks = data["Harvest tanks V01"]

    # Preprocess columns of fermentation recipes
    ferm_recipes["fermentation_prep"] = ferm_recipes['Fermentation__prep (hrs)'] + ferm_recipes['Maintenance__Before prep (hrs)']
    ferm_recipes["fermentation_post"] = ferm_recipes['Fermentation__post (hrs)'] + ferm_recipes['Maintenance__After post (hrs)']
    ferm_recipes["fermentation_time"] = ferm_recipes["Fermentation__process (hrs)"]

    # Preprocess columns of downstream processing recipes
    dsp_recipes['MF'] = dsp_recipes['Name'].str.contains('MF', na=False).astype(int)
    dsp_recipes['Stab'] = dsp_recipes['Stab__In V300'].str.contains('Yes', na=False).astype(int)

    dsp_recipes["UF_fractions"] = dsp_recipes['UF__UF fractions (#)']
    dsp_recipes["FAM/MF_time"] = ((dsp_recipes['Harvest tanks__End weight (kg)'] / dsp_recipes['UF__UF fractions (#)']) /
                                  dsp_recipes['FAM or MF__Process (kg/hr)'])
    dsp_recipes["UF_time"] = ((dsp_recipes['FAM or MF__Weight (kg at F+L)'] / dsp_recipes['UF__UF fractions (#)']) /
                              dsp_recipes["UF__Process (kg/hr)"])

    dsp_recipes["stab_time"] = dsp_recipes['Stab__Process (hrs)']

    # Merge on 'SKU EoF' and 'Fermenter'
    merged_df = dsp_recipes.merge(ferm_recipes, on=['SKU EoF', 'Fermenter'], how='left', suffixes=('y', ''))

    merged_df['fermentation_time'] = merged_df.apply(lambda row: row['Fermentation__process (hrs)'] + row['Broth killing (hrs)'] if row['Broth killing in'] == 'Etanks & Fermenter' else row['Fermentation__process (hrs)'], axis=1)
    merged_df["harvesting_time"] = merged_df.apply(lambda row: row['Harvest tanks__Broth preparation (hrs)'] + row['Broth killing (hrs)'] if row['Broth killing in'] ==
                                        'V01' else row['Harvest tanks__Broth preparation (hrs)'], axis=1)

    # Obtain the minimum minimum weight of the harvest tanks
    min_min_weight = harvest_tanks["Minimum weight (kg)"].min()
    merged_df["min_number_fractions"] = np.floor(min_min_weight / merged_df["Fraction weight (kg)"])
    merged_df['lag_receival_fraction'] = merged_df.apply(get_lag, axis=1)
    print(merged_df["lag_receival_fraction"].tolist())

    # Drop duplicate columns created by the merge
    merged_df = merged_df[['SKU Interm1', 'SKU EoF', 'Enzyme', 'Fermenter', 'Batch weight (kg)', 'Total fractions (kg)',
                           'Fractions (#)', 'fermentation_prep', 'fermentation_time', 'fermentation_post',
                           'harvesting_time', 'FAM/MF_time', 'UF_time', 'stab_time', "UF_fractions",
                           'UF__Weight ccUF (kg)', 'MF', 'Stab', "lag_receival_fraction", "Etanks required (#)"]]

    # Drop NAS
    merged_df = merged_df.dropna()

    # Add key to backtrack enzymes
    # Or, if you want it to start from 1
    merged_df['key'] = range(0, len(merged_df))

    translation_table = merged_df[['key', 'Fermenter', 'SKU EoF', 'SKU Interm1', 'MF', 'Stab']]

    return merged_df, translation_table


def create_product_from_recipe(key: int,
                                     recipe: dict,
                                     translation: dict,
                                     harvesting_tanks: list[int],
                                     v300_tanks: list[int],
                                     UF_tanks: list[int],
                                     flexible_fermenters: bool = True):
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
    if recipe["Fractions (#)"] > 0:
        print(f'Recipe with key {key} has {recipe["Fractions (#)"]} fractions')

        product = create_product_with_fractions(key, recipe, translation,harvesting_tanks, v300_tanks, UF_tanks, flexible_fermenters)

    else:
        product = create_product_batch_fermentation(key, recipe, translation,harvesting_tanks, v300_tanks, UF_tanks, flexible_fermenters)

    return product


def create_product_with_fractions(key: int,
                               recipe: dict,
                               translation: dict,
                               harvesting_tanks: list[int],
                               v300_tanks: list[int],
                               UF_tanks: list[int],
                               flexible_fermenters: bool = True):
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
    MAX_WAIT_FAM_MF = 1
    MAX_WAIT_STAB = 1
    MAX_WAIT_HARVEST = 1
    MAX_WAIT_UF = 1

    # TODO: write a new script for computing the harvesting tank modes
    if recipe["MF"] == True:
        MF = True
    else:
        MF = False

    # These constants are needed to model additional constraints
    constants = {"UF_fractions": recipe["UF_fractions"], "FAM/MF_duration": round(recipe["FAM/MF_time"]),
                 "UF_duration": round(recipe["UF_time"]), "transfer_time_harvesting": 1}

    # The uprod production consist of the following tasks that must be scheduled
    task_names = ["fermentation_prep",
                  "fermentation",
                  "helper_fermenter",
                  "fermentation_post",
                  "receival_fractions",
                  "harvesting",
                  "helper_harvesting",
                  "STAB",
                  "Etanks"
                  ]

    # Get the processing times from the recipe
    processing_times = {"fermentation_prep": recipe["fermentation_prep"],
                        "fermentation_post": recipe["fermentation_post"],
                         "fermentation": recipe["fermentation_time"],
                         "harvesting": recipe["harvesting_time"],
                         "FAM/MF": round(recipe["FAM/MF_time"]),
                         "UF": round(recipe["UF_time"]),
                         "STAB": recipe["stab_time"]
                        }

    # Get the resource modes for the harvesting tasks
    harvesting_translation = {1: "V01_1", 2: "V01_2", 3: "V01_3", 4: "V01_4", 5: "V01_5", 6: "V01_6"}
    harvesting_modes_broth = []

    for j in harvesting_tanks["broth_receival"]:
        harvesting_modes_broth.append([harvesting_translation[i] for i in j])
    harvesting_modes_fractions = []
    for j in harvesting_tanks["fractions_receival"]:
        harvesting_modes_fractions.append([harvesting_translation[i] for i in j])

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
        # TODO: if this is relevant then also implement the E-tanks here
    else:
        # TODO: do this in a more descent way
        fermenter_modes = [recipe["Fermenter"]]

    # TODO: get this from machine files?
    machine_modes = {"fermentation_prep": fermenter_modes,
                    "fermentation_post": fermenter_modes,
                    "fermentation": fermenter_modes,
                    "helper_fermenter": fermenter_modes,
                    "receival_fractions": harvesting_modes_fractions,
                    "harvesting": harvesting_modes_broth,
                    "helper_harvesting": harvesting_modes_broth,
                    "STAB": v300_modes,
                    "F+L": ["F+L_1", "F+L_2", "F+L_3", "F+L_4"],
                    "Etanks": ["E_1", "E_2", "E_3", "E_4"]
                     }

    # Get the task duration from the recipe data
    task_durations = {"fermentation_prep": processing_times["fermentation_prep"],
                          "fermentation_post": processing_times["fermentation_post"],
                          "fermentation": processing_times["fermentation"],
                          "harvesting": processing_times["harvesting"] + constants["transfer_time_harvesting"],
                          "receival_fractions": int(0.5 * processing_times["fermentation"]),
                          "helper_fermenter": processing_times["harvesting"] + constants["transfer_time_harvesting"],
                          "helper_harvesting": int(processing_times["FAM/MF"]/2),
                          "FAM/MF": processing_times["FAM/MF"],
                          "STAB": processing_times["STAB"],
                           "Etanks": processing_times["fermentation"]}

    # Create product
    product = ProductData(key=key, constants=constants, translation=translation)

    id = 0
    for task_name in task_names:
        if task_name in ["helper_fermenter", "receival_fractions", "helper_harvesting", "Etanks"]:
            print(f'Now create a task with fixed duration is false {task_name}')
            fixed_duration = False
        else:
            fixed_duration = True
        task = TaskData(id=id,
                        name=task_name,
                        modes=[(task_durations[task_name], mode) for mode in machine_modes[task_name]],
                        fixed_duration=fixed_duration
                        )
        product.add_task(task)
        id += 1

    print(f'We insert a lag of {recipe["lag_receival_fraction"]}')
    start_before_start_constraints = [("fermentation", "receival_fractions", recipe["lag_receival_fraction"])]
    start_before_end_constraints = [("helper_fermenter", "fermentation", None)]

    end_before_start_constraints = [("fermentation_prep", "fermentation", None),
                                    ("fermentation", "harvesting", None),
                                    ("fermentation", "helper_fermenter", None),
                                    ("helper_fermenter", "fermentation_post", None),
                                    ("harvesting", "helper_harvesting", None)]  # NOWAIT

    end_before_end_constraints = [("harvesting", "helper_fermenter", None)]

    identical_resources = [("fermentation", "helper_fermenter"),
                           ("fermentation", "fermentation_prep"),
                           ("fermentation", "fermentation_post"),
                           ("harvesting", "helper_harvesting")]

    # Handle the fractions for downstream processing
    nr_fracs = constants["UF_fractions"]

    if MF:
        FAM_MF_modes = [(task_durations["FAM/MF"], "MF")]
    else:
        FAM_MF_modes = [(task_durations["FAM/MF"], "FAM_1"), (task_durations["FAM/MF"], "FAM_2"),
                        (task_durations["FAM/MF"], "FAM_3"),
                        (int(task_durations["FAM/MF"] / 2), ["FAM_1", "FAM_2"]),
                        (int(task_durations["FAM/MF"] / 2), ["FAM_2", "FAM_1"]),
                        (int(task_durations["FAM/MF"] / 2), ["FAM_2", "FAM_3"])]

    # For each UF fraction we make a FAM/MF, FAP1, UF and F+L task that requires multiple machines
    UF_modes = [(processing_times["UF"], f"UF_{i}") for i in UF_tanks]
    FAP1_modes = [(processing_times["UF"], f"FAP1_{i}") for i in range(1, 6)]
    FL_modes = [(processing_times["FAM/MF"], f"F+L_{i}") for i in range(1, 5)]

    for uf_frac in range(nr_fracs):
        # TODO: make consistency in task_durations and processing_times dict
        dsp_tasks = [(f"FAM/MF_frac_{uf_frac}", FAM_MF_modes, True),
                     (f"FAP1_frac_{uf_frac}", FAP1_modes, True),
                     (f"UF_frac_{uf_frac}", UF_modes, True),
                     (f"F+L_frac_{uf_frac}", FL_modes, False)]

        # Create task entities and add to product
        for (name, modes, fixed_duration) in dsp_tasks:
            task = TaskData(id=id,
                            name=name,
                            modes=modes,
                            fixed_duration=fixed_duration
                            )
            product.add_task(task)
            id += 1

        start_before_start_extra = [(f"F+L_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", None),
                                    (f"FAM/MF_frac_{uf_frac}", f"F+L_frac_{uf_frac}", None),
                                    (f"FAP1_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                    (f"UF_frac_{uf_frac}", f"FAP1_frac_{uf_frac}", None)
                                    ]

        start_before_end_extra = [(f"UF_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", -MAX_WAIT_UF)]

        end_before_start_extra = [("harvesting", f"FAM/MF_frac_{uf_frac}", None),
                                  (f"FAM/MF_frac_{uf_frac}", f"UF_frac_{uf_frac}", None)
                                  ]

        end_before_end_extra = [(f"FAP1_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                (f"UF_frac_{uf_frac}", f"FAP1_frac_{uf_frac}", None),
                                (f"F+L_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                (f"UF_frac_{uf_frac}", f"F+L_frac_{uf_frac}", None)]

        start_before_start_constraints += start_before_start_extra

        start_before_end_constraints += start_before_end_extra

        end_before_start_constraints += end_before_start_extra

        end_before_end_constraints += end_before_end_extra

    # Fermenter tank in use during fermentation
    start_before_start_constraints += [("helper_harvesting", f"FAM/MF_frac_0", None),
                                       ("harvesting", "helper_fermenter", None),
                                       ("helper_fermenter", "harvesting", None),
                                       ("Etanks", "fermentation", None)]

    start_before_end_constraints += [("STAB", f"UF_frac_{uf_frac}", - MAX_WAIT_STAB),
                                     (f"FAM/MF_frac_0", "harvesting", - MAX_WAIT_FAM_MF)]

    # V300 after last UF fraction
    end_before_start_constraints += [(f"UF_frac_{uf_frac}", "STAB", None)]

    end_before_end_constraints += [(f"FAM/MF_frac_{uf_frac}", "receival_fractions", None),
                                   (f"FAM/MF_frac_{uf_frac}", "helper_harvesting", None),
                                   ("helper_fermenter", "Etanks", None)]


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


def create_product_batch_fermentation(key: int,
                               recipe: dict,
                               translation: dict,
                               harvesting_tanks: list[int],
                               v300_tanks: list[int],
                               UF_tanks: list[int],
                               flexible_fermenters: bool = True):
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
                  "helper_fermenter",
                  "fermentation_post",
                  "STAB",
                  "harvesting",
                  "helper_harvesting"
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

    for j in harvesting_tanks['batch_fermentation']:
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
        # TODO: do this in a more descent way
        fermenter_modes = [recipe["Fermenter"]]

    # TODO: get this from machine files?
    machine_modes = {"fermentation_prep": fermenter_modes,
                    "fermentation_post": fermenter_modes,
                    "fermentation": fermenter_modes,
                    "helper_fermenter": fermenter_modes,
                    "STAB": v300_modes,
                    "F+L": ["F+L_1", "F+L_2", "F+L_3", "F+L_4"],
                     "harvesting": harvesting_modes,
                     "helper_harvesting": harvesting_modes
                     }

    # Get the task duration from the recipe data
    task_durations = {"fermentation_prep": processing_times["fermentation_prep"],
                          "fermentation_post": processing_times["fermentation_post"],
                          "fermentation": processing_times["fermentation"],
                          "harvesting": processing_times["harvesting"],
                          "helper_fermenter": processing_times["harvesting"],
                          "FAM/MF": processing_times["FAM/MF"],
                          "helper_harvesting": int(0.5*processing_times["FAM/MF"]),
                          "F+L": processing_times["FAM/MF"],
                          "STAB": processing_times["STAB"]}

    # These constants are needed to model additional constraints
    constants = {"UF_fractions": recipe["UF_fractions"], "FAM/MF_duration": round(recipe["FAM/MF_time"]),
                            "UF_duration": round(recipe["UF_time"])}

    etanks = ["E_1", "E_2", "E_3", "E_4"]
    print(f'for key {key} Etanks required is {recipe["Etanks required (#)"]}')
    if recipe["Etanks required (#)"] > 0:
        modes_etanks = list(combinations(etanks, int(recipe["Etanks required (#)"])))
        modes_etanks = list(combi for combi in modes_etanks)
        print(f'modes etanks are {modes_etanks}')
        task_names.append("Etanks")
        task_durations["Etanks"] = processing_times["fermentation"]
        machine_modes["Etanks"] = modes_etanks

    # Create product
    product = ProductData(key=key, constants=constants, translation=translation)

    id = 0
    for task_name in task_names:
        if task_name in ["helper_harvesting", "Etanks"]:
            fixed_duration = False
        else:
            fixed_duration = True
        task = TaskData(id=id,
                        name=task_name,
                        modes=[(task_durations[task_name], mode) for mode in machine_modes[task_name]],
                        fixed_duration=fixed_duration
                        )
        product.add_task(task)
        id += 1

    # First, handle the constraints for fermentation and harvesting
    start_before_start_constraints = []

    start_before_end_constraints = [("fermentation", "fermentation_prep", None),
                                    ("fermentation_post", "helper_fermenter", None),
                                    ("harvesting", "fermentation", - MAX_WAIT_HARVEST)]

    end_before_start_constraints = [("fermentation_prep", "fermentation", None),
                                    ("helper_fermenter", "fermentation_post", None),
                                    ("fermentation", "harvesting", None)]

    end_before_end_constraints = []

    identical_resources = [("fermentation_prep", "fermentation"),
                           ("fermentation_post", "fermentation"),
                           ("helper_fermenter", "fermentation"),
                           ("helper_harvesting", "harvesting")]

    # Handle the fractions for downstream processing
    nr_fracs = constants["UF_fractions"]

    if MF:
        FAM_MF_modes = [(task_durations["FAM/MF"], "MF")]
    else:
        FAM_MF_modes = [(task_durations["FAM/MF"], "FAM_1"), (task_durations["FAM/MF"], "FAM_2"),
                        (task_durations["FAM/MF"], "FAM_3"),
                        (int(task_durations["FAM/MF"]/2), ["FAM_1", "FAM_2"]),
                        (int(task_durations["FAM/MF"]/2), ["FAM_2", "FAM_1"]),
                        (int(task_durations["FAM/MF"]/2), ["FAM_2", "FAM_3"])]

    # For each UF fraction we make a FAM/MF, FAP1, UF and F+L task that requires multiple machines
    UF_modes = [(processing_times["UF"], f"UF_{i}") for i in UF_tanks]
    FAP1_modes = [(processing_times["UF"], f"FAP1_{i}") for i in range(1, 6)]
    FL_modes = [(processing_times["FAM/MF"], f"F+L_{i}") for i in range(1, 5)]

    for uf_frac in range(nr_fracs):
        # TODO: make consistency in task_durations and processing_times dict
        dsp_tasks = [(f"FAM/MF_frac_{uf_frac}", FAM_MF_modes, True),
                     (f"FAP1_frac_{uf_frac}", FAP1_modes, True),
                     (f"UF_frac_{uf_frac}", UF_modes, True),
                     (f"F+L_frac_{uf_frac}", FL_modes, False)]

        # Create task entities and add to product
        for (name, modes, fixed_duration) in dsp_tasks:

            task = TaskData(id=id,
                            name=name,
                            modes=modes,
                            fixed_duration=fixed_duration
                            )
            product.add_task(task)
            id += 1

        start_before_start_extra = [(f"F+L_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", None),
                                    (f"FAM/MF_frac_{uf_frac}", f"F+L_frac_{uf_frac}", None),
                                    (f"FAP1_frac_{uf_frac}", f"UF_frac_{uf_frac}", None),
                                    (f"UF_frac_{uf_frac}", f"FAP1_frac_{uf_frac}", None)]

        start_before_end_extra = [(f"UF_frac_{uf_frac}", f"FAM/MF_frac_{uf_frac}", -MAX_WAIT_UF)]

        end_before_start_extra = [("harvesting", f"FAM/MF_frac_{uf_frac}", None),
                                 (f"FAM/MF_frac_{uf_frac}", f"UF_frac_{uf_frac}", None)]

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
    end_before_end_constraints += [(f"FAM/MF_frac_{nr_fracs - 1}", "helper_harvesting", None)]

    # Fermenter tank in use during harvesting
    start_before_start_constraints += [("helper_harvesting", f"FAM/MF_frac_0", None),
                           ("harvesting", "helper_fermenter", None),
                           ("helper_fermenter", "harvesting", None)]

    start_before_end_constraints += [("STAB", f"UF_frac_{uf_frac}", - MAX_WAIT_STAB),
                                     (f"FAM/MF_frac_0", "harvesting", - MAX_WAIT_FAM_MF)]

    # V300 after last UF fraction
    end_before_start_constraints += [(f"UF_frac_{uf_frac}", "STAB", None)]

    # Temporal constraints that only apply to the final one or first one
    # Ordering of the fractions
    for frac in range(nr_fracs - 1):
        start_before_start_constraints += [(f"UF_frac_{frac}", f"UF_frac_{frac + 1}", None)]

    if recipe["Etanks required (#)"] > 0:
        start_before_start_constraints += [("Etanks", "fermentation", None)]
        end_before_end_constraints += [("fermentation", "Etanks", None)]

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


