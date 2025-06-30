import pandas as pd

df = pd.read_csv('results_hypothesis_2_larger_timelimit.csv')

print(df)

solver_agg = df.groupby('solver')['runtime'].mean().reset_index()
solver_agg['group_by'] = 'solver'

resources_agg = df.groupby('resources')['runtime'].mean().reset_index()
resources_agg['group_by'] = 'resources'

# Standardize column names
solver_agg.columns = ['value', 'mean_runtime', 'group_by']
resources_agg.columns = ['value', 'mean_runtime', 'group_by']

combined = pd.concat([solver_agg, resources_agg], ignore_index=True)

print(combined)
print("\n")

resources_agg = df.groupby(['size', 'solver', 'resources'])['runtime'].mean().reset_index()
print(resources_agg)
