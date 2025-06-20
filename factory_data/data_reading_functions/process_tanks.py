import pandas as pd
from itertools import combinations


def remove_rendundant_sets(sets):
    # Sort by length descending
    sets = [set(x) for x in sets]
    sets.sort(key=lambda s: -len(s))

    # Copy the list to work on
    filtered = sets.copy()

    for current in sets:
        # Check if any other set in the list is a proper subset of `current`
        if any(other < current for other in filtered if other != current):
            filtered.remove(current)

    # Optionally convert back to sorted tuples
    result = [tuple(sorted(s)) for s in filtered]

    return result


def get_modes_v300(recipes):
    """
    This function takes the enzyme recipes and reads the data about the V300 preferences and the harvesting tank
    capacity and translates this in to possible sets of tanks that can do the V300 operation of a specific enzyme.

    Args:
        recipes (dict): A dictionary with all recipe data.

    Returns:
        dict: A dictionary with per enzyme all possible sets of tanks that can do the V300 (stab) operation.
    """
    # FILE LOCATION
    dir = 'factory_data/seclin_recipes'

    # Read in V300 tank capacity
    capacity = pd.read_csv(f'{dir}/file_D_V300 tanks.csv', index_col=False)
    capacity_dict = dict(zip(capacity['ID'], capacity['Capacity (kg)']))

    # TODO: obtain the number of V300 tanks

    # Read in V01 tank preferences
    preferences = pd.read_csv(f'{dir}/file_D_V300 preferences.csv')

    # Read in SKU translation
    skus = pd.read_csv(f'{dir}/file_D_DEF_Enzymes.csv')

    # Merge side by side of preferences and skus
    preferences_per_sku = pd.concat([skus, preferences], axis=1)
    # TODO: automate this based on the number of V300 tanks
    preferences_dict = {
        row["SKU EoF"]: [row["1"], row["2"], row["3"], row["4"], row["5"], row["6"], row["7"], row["8"], row["9"],
                         row["10"], row["11"]]
        for _, row in preferences_per_sku.iterrows()
    }

    # Map out all possible combinations of machines
    # For each combination compute the total capacity
    # TODO: automate this based on the number of V300 tanks
    tanks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    num_tanks = len(tanks)
    all_combinations = []

    for r in range(1, len(tanks) + 1):
        combs = list(combinations(tanks, r))
        all_combinations.extend(combs)

    filtered_combinations = {}
    for length in range(1, num_tanks + 1):
        filtered_combinations[length] = [comb for comb in all_combinations if len(comb) == length]

    # Print or use the result
    comb_weights = {}
    for comb in all_combinations:
        total_capacity = 0
        for i in comb:
            total_capacity += capacity_dict[i]
        comb_weights[comb] = total_capacity

    allowed_modes_harvesting = {}
    # Iterate through EOFs
    for enz in recipes.keys():
        # Obtain batch weight
        batch_weight = recipes[enz]['UF__Weight ccUF (kg)']

        # Filter only the combinations that have the capacity constraint satisfied
        allowed_combs = [k for k, v in comb_weights.items() if v >= batch_weight]

        # Obtain which tanks cannot be used for this EOF
        sku = enz.split("_")[0]
        allowed_tanks = preferences_dict[int(sku)]
        not_allowed_tanks = [i + 1 for i, val in enumerate(allowed_tanks) if val == 0]

        # Filter these combinations out of the list of combinations
        if len(not_allowed_tanks) > 0:
            allowed_combs = [t for t in allowed_combs if not any(i in not_allowed_tanks for i in t)]

        # Remove redundant sets
        allowed_combs = remove_rendundant_sets(allowed_combs)
        allowed_modes_harvesting[enz] = allowed_combs

    return allowed_modes_harvesting


def get_modes_harvesting(recipes: dict) -> dict:
    """
    This function takes the enzyme recipes and reads the data about the harvesting preferences and the harvesting tank
    capacity and translates this in to possible sets of tanks that can do the harvesting operation of a specific enzyme.

    Args:
        recipes (dict): A dictionary with all recipe data.

    Returns:
        dict: A dictionary with per enzyme all possible sets of tanks that can do the harvesting operation.
    """
    # FILE LOCATION
    dir = 'factory_data/seclin_recipes'

    # Read in V01 tank capacity
    capacity = pd.read_csv(f'{dir}/file_d_Harvest tanks V01.csv')

    capacity_dict = dict(zip(capacity['ID'], capacity['Capacity (kg)']))
    # Read in V01 tank preferences
    preferences = pd.read_csv(f'{dir}/file_D_Harvesting preferences V01.csv')

    # Read in SKU translation
    skus = pd.read_csv(f'{dir}/file_D_DEF_Enzymes.csv')

    # Merge side by side of preferences and skus
    preferences_per_sku = pd.concat([skus, preferences], axis=1)
    preferences_dict = {
        row["SKU EoF"]: [row["1"], row["2"], row["3"], row["4"], row["5"], row["6"]]
        for _, row in preferences_per_sku.iterrows()
    }

    # Map out all possible combinations of machines
    # For each combination compute the total capacity
    # TODO: automate this based on the harvesting tank files and automate reading the number of tanks throughout this
    #  script
    tanks = [1, 2, 3, 4, 5, 6]
    num_tanks = len(tanks)
    all_combinations = []

    for r in range(1, len(tanks) + 1):
        combs = list(combinations(tanks, r))
        all_combinations.extend(combs)

    filtered_combinations = {}
    for length in range(1, num_tanks + 1):
        filtered_combinations[length] = [comb for comb in all_combinations if len(comb) == length]

    # Print or use the result
    comb_weights = {}
    for comb in all_combinations:
        total_capacity = 0
        for i in comb:
            total_capacity += capacity_dict[i]
        comb_weights[comb] = total_capacity

    allowed_modes_harvesting = {}
    # Iterate through EOFs
    for enz in recipes.keys():
        # Obtain batch weight
        batch_weight = recipes[enz]['Batch weight (kg)']

        # Filter only the combinations that have the capacity constraint satisfied
        allowed_combs = [k for k, v in comb_weights.items() if v >= batch_weight]

        # Obtain which tanks cannot be used for this EOF
        sku = enz.split("_")[0]
        allowed_tanks = preferences_dict[int(sku)]
        not_allowed_tanks = [i + 1 for i, val in enumerate(allowed_tanks) if val == 0]

        # Filter these combinations out of the list of combinations
        if len(not_allowed_tanks) > 0:
            allowed_combs = [t for t in allowed_combs if not any(i in not_allowed_tanks for i in t)]

        # Remove redundant sets
        allowed_combs = remove_rendundant_sets(allowed_combs)
        allowed_modes_harvesting[enz] = allowed_combs

    return allowed_modes_harvesting

