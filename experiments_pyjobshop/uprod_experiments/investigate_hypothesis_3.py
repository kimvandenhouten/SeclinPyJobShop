from src import CPModel
from src.entities.instance import Instance
import json
import pandas as pd

TIME_LIMIT = 600
# Load from file
data = []
for size in [5]:
    for solver in ["ortools", "cpoptimizer"]:
        for ADD_SETUPS in [False]:
            print(f'ADD SETUPs {ADD_SETUPS}')

            file_name = f"factory_data/uprod_instances/instance_size_{size}_sequence_dependent_setup.json"

            # Read the JSON string from the file
            with open(file_name, "r") as file:
                json_string = file.read()

            # Parse the JSON string to a Python dictionary
            instance_data = json.loads(json_string)

            # Convert back to Instance object
            instance = Instance.from_dict(instance_data)

            # Create PyJobshop CP Model
            print(f'open instance from file {file_name}')
            model = CPModel(instance)

            if ADD_SETUPS:
                print(f'add setups')
                model.add_set_up_times()

            result = model.solve(solver=solver,
                                 time_limit=TIME_LIMIT,
                                 display=False,
                                 print_result=True,
                                 print_sol=False,
                                 output_file=f"plots/instance_size_{size}.png")

            data.append(
                {"solver": solver, "status": result.status, "objective": result.objective, "runtime": result.runtime,
                 "setups": ADD_SETUPS, "size": size, "lb": result.lower_bound})

            data_df = pd.DataFrame(data)
            data_df.to_csv(f"results_hypothesis_3_larger_timelimit_.csv")
