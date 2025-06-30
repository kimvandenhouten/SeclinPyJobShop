"""
Code inspired by:
https://pyjobshop.org/latest/examples/project_scheduling.html

This code is used to replace the CP model used in the AAAI paper with a PyJobShop variants.
If we would continue with this code, improvements can be in the direction of:
- Directly using the objects from PyJobShop to do the resource chaining (now there is ugly step to a dict)
- Directly using the constraints from the PyJobShop model to construct the constraints in the Temporal Network
  (now this is coded in rcpsp_max/temporal_networks/stnu_rcpsp_max.py)
"""

from pyjobshop import Model
# Import
from rcpsp_max.solvers.RCPSP_CP_benchmark import RCPSP_CP_Benchmark
from PyJobShopIntegration.utils import (rte_data_to_pyjobshop_solution, sample_for_rte, get_resource_chains,
                                        add_resource_chains)
from PyJobShopIntegration.PyJobShopSTNU import PyJobShopSTNU
from PyJobShopIntegration.Sampler import DiscreteUniformSampler

from pyjobshop.Model import Model
from pyjobshop.plot import plot_task_gantt, plot_resource_usage

from temporal_networks.cstnu_tool.stnu_to_xml_function import stnu_to_xml
from temporal_networks.cstnu_tool.call_java_cstnu_tool import run_dc_algorithm
from temporal_networks.stnu import STNU
from temporal_networks.rte_star import rte_star

import numpy as np
from general.logger import get_logger
import matplotlib.pyplot as plt
from typing import NamedTuple

logger = get_logger(__name__)


# Following PyJobShop example
class Mode(NamedTuple):
    job: int
    duration: int
    demands: list[int]


# GENERAL SETTINGS
SEED = 1
DIRECTORY_INSTANCES = 'rcpsp_max/data'
instance_folder = "j10"
instance_id = 2
instance_sub_id = 2
noise_factor = 1

# Data parsching (this originates from the AAAI repo)
instance = RCPSP_CP_Benchmark.parsche_file(DIRECTORY_INSTANCES, instance_folder, instance_id, noise_factor)

# MODEL
model = Model()

# It's not necessary to define jobs, but it will add coloring to the plot.
jobs = [model.add_job() for _ in range(instance.num_tasks)]
tasks = [model.add_task(job=jobs[idx]) for idx in range(instance.num_tasks)]

# RCPSP uses renewable resources
resources = [model.add_renewable(capacity) for capacity in instance.capacity]

for idx in range(instance.num_tasks):
    model.add_mode(tasks[idx], resources, instance.durations[idx], instance.needs[idx])

for (pred, lag, suc) in instance.temporal_constraints:
    model.add_start_before_start(tasks[pred], tasks[suc], delay=lag)

# Solve model and get resource chains
result = model.solve(time_limit=5, display=False)
solution = result.best


### HERE STARTS OUR CODE ###
# Define the stochastic processing time distributions
duration_distributions = DiscreteUniformSampler(lower_bounds=np.random.randint(1, 3, len(model.tasks)),
                                 upper_bounds=np.random.randint(5, 8, len(model.tasks)))
# Create stnu from concrete model
stnu = PyJobShopSTNU.from_concrete_model(model, duration_distributions)

# TODO: the get_resource_chains does not work yet for multi-mode, and also in general, we have nothing for models
#       with mixed renewable resources and machines
schedule_pyjobshop = [{"task": i, "start": task.start, "end": task.end} for i, task in enumerate(result.best.tasks)]
resource_chains, resource_assignments = get_resource_chains(schedule_pyjobshop, instance.capacity, instance.needs,
                                                            complete=True)
stnu = add_resource_chains(stnu, resource_chains)

# Write stnu to xml for DC-checking
file_name = f"example_rcpsp_max_pyjobshop_stnu"
stnu_to_xml(stnu, file_name, "temporal_networks/cstnu_tool/xml_files")

# Run the DC algorithm using the Java CSTNU tool, the result is written to a xml file
dc, output_location = run_dc_algorithm("temporal_networks/cstnu_tool/xml_files", file_name)

if dc:
    logger.info(f'The network resulting from the PyJobShop solution is DC')
else:
    logger.info(f'The network is not DC')

if dc:
    # TODO: we could have some sort of Simulator/Evaluator class to do all of this
    # Read ESTNU xml file into Python object that was the output from the previous step
    estnu = STNU.from_graphml(output_location)
    sample_duration = duration_distributions.sample()
    sample = sample_for_rte(sample_duration, estnu)  # TODO: this could then be integrated in a Simulator Class
    logger.debug(f'Sample dict that will be given to RTE star is {sample_duration}')

    # Run RTE algorithm with sample
    rte_data = rte_star(estnu, oracle="sample", sample=sample)

    # Convert to PyJobShop solution for visualization
    ## TODO: currently objective is not overwritten in Solution object
    simulated_solution, objective = rte_data_to_pyjobshop_solution(solution, estnu, rte_data, len(model.tasks), "makespan")
    logger.info(f'Simulated solution has objective {objective}')

    # Plotting
    data = model.data()
    fig, axes = plt.subplots(
        data.num_resources + 1,
        figsize=(12, 16),
        gridspec_kw={"height_ratios": [6] + [1] * data.num_resources},
    )

    plot_task_gantt(result.best, model.data(), ax=axes[0])
    plot_resource_usage(result.best, model.data(), axes=axes[1:])
    plt.savefig('PyJobShopIntegration/images/rcpsp_max_example.png')