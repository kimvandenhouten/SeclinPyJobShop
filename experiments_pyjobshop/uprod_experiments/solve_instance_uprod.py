from src import CPModel
from src.entities.instance import Instance
import json

# Load from file
for instance_name in ["plan_july"]:

    filename = f"factory_data/uprod_instances/instance_{instance_name}.json"
    # Read the JSON string from the file
    with open(filename, "r") as file:
        json_string = file.read()

    # Parse the JSON string to a Python dictionary
    data = json.loads(json_string)

    # Convert back to Instance object
    instance = Instance.from_dict(data)

    # Create PyJobshop CP Model
    print(f'open instance from file {instance_name}')
    model = CPModel(instance)

    model.add_set_up_times()
    status = model.solve(solver='cpoptimizer',
                         time_limit=30,
                         display=True,
                         print_result=True,
                         print_sol=True,
                         output_file=f"plots/instance_{instance_name}.png")
