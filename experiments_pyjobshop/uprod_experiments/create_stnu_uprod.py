import copy

from src import CPModel
from src.entities.instance import Instance
import json
import numpy as np
from general.logger import get_logger

logger = get_logger(__name__)

from PyJobShopIntegration.utils import sample_for_rte, rte_data_to_pyjobshop_solution
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.rte_star import rte_star
from temporal_networks.stnu import STNU

from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler
from PyJobShopIntegration.parser import create_instance
from pyjobshop.plot import plot_machine_gantt

import numpy as np
import general.logger
import matplotlib.pyplot as plt


# Load from file
for key in [20]:
    file_name = f"factory_data/uprod_instances/instance_size_{key}.json"

    # Read the JSON string from the file
    with open(file_name, "r") as file:
        json_string = file.read()

    # Parse the JSON string to a Python dictionary
    data = json.loads(json_string)

    # Convert back to Instance object
    instance = Instance.from_dict(data)

    print(f'product ids')
    print(instance.product_ids)



    # Create PyJobshop CP Model
    print(f'open instance from file {file_name}')
    model = CPModel(instance)

    #model.add_set_up_times()
    result = model.solve(solver='cpoptimizer',
                         time_limit=30,
                         display=True,
                         print_result=True,
                         print_sol=False,
                         output_file=f"plots/instance_size_debug.png")

    plot_machine_gantt(result.best, model.model.data(), plot_labels=False)
    plt.savefig(f'partial_order_schedule_{key}.png')
    plt.close()

    names = []
    for task in model.model.tasks:
        names.append(task.name)
    durations = []
    for task in result.best.tasks:
        duration = model.model.modes[task.mode].duration
        durations.append(duration)
    print(names)
    print(durations)

    fixed_duration_false = ['V01_during_FAM'] + [f'F+L_frac_{i}' for i in range(0, 15)]

    lower_bounds = []
    upper_bounds = []
    for i, name in enumerate(names):
        if name not in fixed_duration_false:
            lb = durations[i]
            ub = durations[i]
        else:
            lb = 0
            ub = 99999
        if name == "fermentation":
            lb = durations[i]
            ub = durations[i] + 50
            print(f'we now create the links for fermentation')
        lower_bounds.append(lb)
        upper_bounds.append(ub)

    duration_distributions = DiscreteUniformSampler(lower_bounds, upper_bounds)

    stnu = PyJobShopSTNU.from_concrete_model(model.model, duration_distributions=duration_distributions)

    print(f'STNU created')
    name_graph = "seclin_STNU"
    dir_stnus = "temporal_networks/cstnu_tool/xml_files"

    # Add resource chains from solution to the stnu
    stnu.add_resource_chains(result.best, model.model)

    # Write stnu to xml which is required for using Java CSTNU tool algorithms
    stnu_to_xml(stnu, name_graph, dir_stnus)

    print(f'Resource chains added')

    # Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
    dc, output_location = run_dc_algorithm(dir_stnus, name_graph)

    if dc:
        logger.info(f'The network resulting from the PyJobShop solution is DC')
    else:
        logger.info(f'The network is not DC')

    if dc:
        objectives = []
        for num_sim in range(100):
            print(f'START NEW SAMPLE {num_sim}')
            # TODO: we could have some sort of Simulator/Evaluator class to do all of this
            # Read ESTNU xml file into Python object that was the output from the previous step
            estnu = STNU.from_graphml(output_location)
            sample_duration = duration_distributions.sample()
            sample = sample_for_rte(sample_duration, estnu)  # TODO: this could then be integrated in a Simulator Class
            #logger.info(f'Sample dict that will be given to RTE star is {sample_duration}')

            # Run RTE algorithm with sample
            rte_data = rte_star(estnu, oracle="sample", sample=sample)

            # Convert to PyJobShop solution for visualization
            simulated_solution, objective = rte_data_to_pyjobshop_solution(result.best, estnu, rte_data, len(model.model.tasks),
                                                                           "makespan")
            logger.info(f'Simulated solution has objective {objective}')
            #plot_machine_gantt(simulated_solution, model.model.data(), plot_labels=False)
            #plt.savefig(f'STNU_gantt_{key}_{num_sim}.png')

            objectives.append(objective)
        # Plot histogram

            plt.hist(objectives, bins=range(min(objectives), max(objectives) + 2), edgecolor='black', align='left')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of Objectives')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig(f'Histogram of objectives.png')
            plt.close()

