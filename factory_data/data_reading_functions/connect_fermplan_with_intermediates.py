import pandas as pd

DIR = "factory_data/seclin_recipes"
ferm_plan_path = f"factory_data/fermentation_plans/plan_july_2024.csv"
interm1_path = f"{DIR}/file_D_Intermediate 1.csv"
ferm_recipes_path = f"{DIR}/Recipes fermentation.xlsx"
dsp_recipes_path = f"{DIR}/Recipes DSP.xlsx"

# TODO: import Ferm plan
ferm_plan = pd.read_csv(ferm_plan_path, delimiter=";")
ferm_plan = ferm_plan[['SKU EoF', 'Fermenter', 'MF (%)', 'Stab (%)']]
ferm_plan['MF'] = ferm_plan['MF (%)'].replace(100, 1)
ferm_plan['Stab'] = ferm_plan['MF (%)'].replace(100, 1)

print(ferm_plan['MF'].tolist())


# TODO: import Interm1 definitions
interms = pd.read_csv(interm1_path)
interms = interms[['SKU Interm1', 'SKU EoF', "MF", "Stab"]]

# TODO: merge interms to ferm_plan
merged_plan = pd.merge(ferm_plan, interms, how='left', on=['SKU EoF', 'MF', 'Stab'])

print(merged_plan)

# TODO: import fermentation recipes
ferm_recipes = pd.read_excel(ferm_recipes_path)
ferm_recipes = ferm_recipes[["SKU EoF", "Fermenter"]]

# TODO: import DSP recipes
dsp_recipes = pd.read_excel(dsp_recipes_path)
dsp_recipes = dsp_recipes[["SKU Interm1", "Fermenter", "SKU EoF"]]

# TODO: import interm 1 - Ferm combinations --> Create IDs
print(dsp_recipes_path)
# TODO: connect ids to ferm plans



