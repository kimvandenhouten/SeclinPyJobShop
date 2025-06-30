import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

fitting = False

# Load data
df = pd.read_csv("factory_data/fermentation_times/ferm_times.csv", delimiter=";")

# Convert 'tferm (hrs)' to float
df['tferm (hrs)'] = df['tferm (hrs)'].astype(str).str.replace(',', '.', regex=False).astype(float)

# We now merge the ferm times that were execution on V200 and V218 because they come from the same recipe
df['Machine'] = df['Machine'].replace({'F200': 'F218'})


# Create Machine_PROD column
df['Machine_PROD'] = df['Machine'] + ' - ' + df['PROD']

# Count the occurrences of each unique Machine_PROD value
counts = df['Machine_PROD'].value_counts()

# Display the counts
print(counts)

# Get statistics on the counts
print("\nStatistics on the counts:")
print(f"Average: {counts.mean():.2f}")
print(f"Min: {counts.min()}")
print(f"Max: {counts.max()}")
print(f"Std: {counts.std():.2f}")
print(f"Total unique combinations: {counts.count()}")




import pandas as pd

df_recipes = pd.read_excel('factory_data/seclin_recipes/Recipes fermentation.xlsx')
print(df_recipes.columns)
df_recipes['Fermenter'] = df_recipes['Fermenter'].replace({'V42': 'F42',
                                           'V100': 'F100',
                                           'V140': 'F140',
                                           'V200': 'F218',
                                           'V218': 'F218'})


df_recipes['Machine_PROD'] = df_recipes['Fermenter'] + ' - ' + df_recipes['Enzyme']

df_recipes = df_recipes[['Machine_PROD',  'Fermentation__process (hrs)', 'Fermentation__process extra (hrs)', 'Fermentation__post (hrs)']]

# Some stats about the times
ferm_stats = df.groupby('Machine_PROD')['tferm (hrs)'].agg(['count', 'min', 'mean', 'max', 'std'])
print(ferm_stats)


merged_df = pd.merge(ferm_stats, df_recipes, on='Machine_PROD')

merged_df.to_csv('factory_data/fermentation_times/fermentation_times_analysis.csv', index=False)