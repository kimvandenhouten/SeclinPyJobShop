import pandas as pd

# Load CSV without parse_dates
from factory_data.data_reading_functions.read_data_files import load_dataframes_from_files
import os

# Load CSV
df = pd.read_csv("factory_data/csv_inputs/Log_ResourceUse.csv")
dir_path = "factory_data/csv_inputs"
filenames = os.listdir(dir_path)
print(f'file names are {filenames}')
data = load_dataframes_from_files(dir_path, filenames)

# Convert Excel serial numbers to datetime
#for col in ["Claim time", "Release time", "Start use", "End use"]:
    #df[col] = pd.to_datetime(df[col], unit='D', origin='1899-12-30')

# Now you can calculate relative hours
min_time = df["Claim time"].min()
df["Claim time (hrs)"] = (df["Claim time"] - min_time)
df["Release time (hrs)"] = (df["Release time"] - min_time)

from factory_data.data_reading_functions.process_tanks import read_equipment_seclin

tanks = read_equipment_seclin(data)

df["Equipment"] = pd.Categorical(df["Equipment"], categories=tanks, ordered=True)
df = df.sort_values('Equipment', ascending=False)

df["Batch ID"] = df["Batch ID"].astype(int)

import matplotlib.pyplot as plt

# Assign a unique color per batch
colors = {str(batch): f"C{i}" for i, batch in enumerate(sorted(df["Batch ID"].unique()))}
print(colors)
fig, ax = plt.subplots(figsize=(10, 6))

# Track which batch labels have already been added to the legend
already_labeled = set()

for i, row in df.iterrows():
    batch = row["Batch ID"]
    label = batch if batch not in already_labeled else None
    ax.barh(
        row["Equipment"],
        row["Release time (hrs)"] - row["Claim time (hrs)"],
        left=row["Claim time (hrs)"],
        color=colors[str(batch)],
        label=label
    )
    already_labeled.add(batch)

# Add the legend once
handles, labels = ax.get_legend_handles_labels()
sorted_pairs = sorted(zip(labels, handles), key=lambda x: int(x[0]))
sorted_labels, sorted_handles = zip(*sorted_pairs)
ax.legend(sorted_handles, sorted_labels, title="Batch ID", bbox_to_anchor=(1.05, 1), loc='upper left')


# Labels and layout
ax.set_xlabel("Time (hrs)")
ax.set_ylabel("Equipment")
ax.set_title("Gantt Chart of Equipment Usage")
plt.tight_layout()
plt.savefig("gannt")

