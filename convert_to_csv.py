import os
import pandas as pd

dir_path = "factory_data/InputData/"
filenames = os.listdir(dir_path)
print(f'file names are {filenames}')

for file_name in filenames:
    df = pd.read_excel(os.path.join(dir_path, file_name))
    df.to_csv(f"factory_data/csv_inputs/{file_name[0:-5]}.csv", index=False)