from src import CPModel
from src.entities.instance import Instance
import json

# Load from file
for key in [10]:
    file_name = f"factory_data/uprod_instances/instance_size_{key}.json"

    # Read the JSON string from the file
    with open(file_name, "r") as file:
        json_string = file.read()

    # Parse the JSON string to a Python dictionary
    data = json.loads(json_string)

    # Convert back to Instance object
    instance = Instance.from_dict(data)

    # Create PyJobshop CP Model
    print(f'open instance from file {file_name}')
    model = CPModel(instance)
    #model.add_set_up_times()
    status = model.solve(solver='cpoptimizer',
                         time_limit=3600*10,
                         display=True,
                         print_result=True,
                         print_sol=False,
                         output_file=f"plots/instance_size_{key}.png")
