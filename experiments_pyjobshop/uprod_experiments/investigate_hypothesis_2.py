import pandas as pd

from src import CPModelRenewableResources, CPModel
from src.entities.instance import Instance
import json
data = []

TIME_LIMIT = 60
# Load from file
for size in [1, 5, 10, 20]:
    for instance_id in range(0, 5):
        file_name = f"factory_data/uprod_instances/instance_size_{size}_{instance_id}.json"

        # Read the JSON string from the file
        with open(file_name, "r") as file:
            json_string = file.read()

        # Parse the JSON string to a Python dictionary
        instance_data = json.loads(json_string)

        # Convert back to Instance object
        print(f'open instance from file {file_name}')
        instance = Instance.from_dict(instance_data)

        for solver in ["cpoptimizer", "ortools"]:
            # Create PyJobshop CP Model with Renwable resources

            model = CPModelRenewableResources(instance)

            print(f'\nSolve with {solver} and renewable resources')
            result = model.solve(solver=solver,
                                 time_limit=TIME_LIMIT,
                                 display=False,
                                 print_result=True,
                                 print_sol=False,
                                 location_gantt=f"plots/instance_{size}_{instance_id}_renewable_gantt.png")
            data.append({"solver": solver, "status": result.status, "objective": result.objective, "runtime": result.runtime,
                         "resources": "renewable", "size": size, "instance_id": instance_id})

            data_df = pd.DataFrame(data)
            data_df.to_csv(f"results_hypothesis_2_larger_timelimit.csv")

            # Create PyJobshop CP Model with Machine Modes
            print(f'\nSolve with {solver} and machine modes')
            model = CPModel(instance)

            # model.add_set_up_times()
            result = model.solve(solver=solver,
                                 time_limit=TIME_LIMIT,
                                 display=False,
                                 print_result=True,
                                 print_sol=False,
                                 output_file=f"plots/instance_{size}_{instance_id}.png")
            data.append(
                {"solver": solver, "status": result.status, "objective": result.objective, "runtime": result.runtime,
                 "resources": "flexible_machines", "size": size, "instance_id": instance_id})

            data_df = pd.DataFrame(data)
            data_df.to_csv(f"results_hypothesis_2_larger_timelimit.csv")
