import pandas as pd
import numpy as np

# Read input data
ferm_plan = pd.read_excel("factory_data/seclin_recipes/Fermentation plan.xlsx")
dsp_recipes = pd.read_excel("factory_data/seclin_recipes/Recipes DSP.xlsx")
ferm_recipes = pd.read_excel("factory_data/seclin_recipes/Recipes fermentation.xlsx")


# Ferm recipes filtering
#ferm_recipes = ferm_recipes[ferm_recipes["Fractions (#)"] == 0]
#ferm_recipes = ferm_recipes[ferm_recipes["Etanks required (#)"] == 0]
ferm_recipes["V01 tanks (#)"] = (np.minimum(ferm_recipes["V01 group1 (kg)"], 1) +
                                 np.minimum(ferm_recipes["V01 group2 (kg)"], 1) +
                                 np.minimum(ferm_recipes["V01 group3 (kg)"], 1) +
                                 np.minimum(ferm_recipes["V01 group4 (kg)"], 1))
print(ferm_recipes.columns)
print(f'Batch weights ferm recipes')
print(ferm_recipes['Batch weight (kg)'].tolist())


# DSP recipes filter
dsp_recipes = dsp_recipes[dsp_recipes["Broth killing in"] == 'V01']
dsp_recipes = dsp_recipes[dsp_recipes['Stab__In V300'] == 'Yes']
#dsp_recipes = dsp_recipes[dsp_recipes['UF__UF fractions (#)'] = 2]

# Create new columns for processing times
ferm_recipes["fermentation_prep"] = ferm_recipes['Fermentation__prep (hrs)'] + ferm_recipes['Maintenance__Before prep (hrs)']
ferm_recipes["fermentation_post"] = ferm_recipes['Fermentation__post (hrs)'] + ferm_recipes['Maintenance__After post (hrs)']
ferm_recipes["fermentation_time"] = ferm_recipes["Fermentation__process (hrs)"]
dsp_recipes["harvesting_time"] = dsp_recipes["Broth killing (hrs)"] + dsp_recipes["Harvest tanks__Broth preparation (hrs)"]
dsp_recipes["FAM/MF_time"] = (dsp_recipes['Harvest tanks__End weight (kg)'] / dsp_recipes['UF__UF fractions (#)']) / dsp_recipes['FAM or MF__Process (kg/hr)']
dsp_recipes["UF_time"] = (dsp_recipes['FAM or MF__Weight (kg at F+L)'] / dsp_recipes['UF__UF fractions (#)']) / dsp_recipes["UF__Process (kg/hr)"]
dsp_recipes["stab_time"] = dsp_recipes['Stab__Process (hrs)']
dsp_recipes["UF_fractions"] = dsp_recipes['UF__UF fractions (#)']

print(f'batch weights from dsp recipes')
print(dsp_recipes['Batch weight (kg)'].tolist())

# Now how are we gonna couple this?
print(f'SKUs from ferm {ferm_recipes["SKU EoF"].tolist()}')
print(f'SKUs from DSP {dsp_recipes["SKU EoF"].tolist()}')
print(f'SKUs Interm1 from DSP {dsp_recipes["SKU Interm1"].tolist()}')

# Merge on 'SKU EoF' and 'Fermenter'
merged_df = dsp_recipes.merge(ferm_recipes, on=['SKU EoF', 'Fermenter'], how='left', suffixes=('y', ''))

# Drop duplicate columns created by the merge
#merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_y')]
merged_df = merged_df[['SKU Interm1', 'SKU EoF', 'Fermenter', 'Batch weight (kg)', 'fermentation_prep', 'fermentation_time',
                       'fermentation_post', 'harvesting_time', 'FAM/MF_time', 'UF_time', 'stab_time',
                       "UF_fractions", 'UF__Weight ccUF (kg)']]

# Save or display the merged DataFrame
print(merged_df.head())
# Remove rows with any NaN value
merged_df = merged_df.dropna()
print(f'Batch weights after merge')
print(merged_df["Batch weight (kg)"].tolist())
merged_df.to_csv('merged_recipes.csv')




