from pyjobshop import Model
import matplotlib.pyplot as plt
from pyjobshop import Model
import random
model = Model()
model.set_objective(weight_max_tardiness=1, weight_makespan=1)
# Add all the machines to the model
machines_trans = ["plate_filter", "tank", "packaging_line"]
machines_capacity = [4, 19, 3]
machines = []
machines_dict = {}
for i, name in enumerate(machines_trans):
    machines_dict[name] = []
    for j in range(machines_capacity[i]):
        machine = (model.add_machine(name=f'{name}_{j}'))
        machines_dict[name].append(machine)
        machines.append(machine)

print(f'All machines are added')
# Start making jobs for the products
nr_products = 20
jobs = [model.add_job(due_date=random.randint(10, 30)) for _ in range(nr_products)]
job_tasks = {}
for j, job in enumerate(jobs):

    tasks = []
    tasks_dict = {}
    names = ["preparation", "filtering", "reception", "formulation", "packaging"]
    for name in names:
        if name == "receiving_from_filter":
            task = model.add_task(job=job, name=name, fixed_duration=False)
        else:
            task = model.add_task(job=job, name=name)
        tasks.append(task)
        tasks_dict[name] = task

    print(f'For a new product all tasks are added')

    job_tasks[j] = tasks_dict

    # Fix all the constraints for one product
    processing_times = {"preparation": random.randint(5, 15),
                        "filtering": random.randint(4, 7),
                        "formulation": random.randint(2, 5),
                        "packaging": random.randint(2, 5)}

    machine_requirements = {"preparation": "tank",
                            "filtering": "plate_filter",
                            "reception": "tank",
                            "formulation": "tank",
                            "packaging": "packaging_line"
                             }

    task_durations = {"preparation": processing_times["preparation"],
                        "filtering": processing_times["filtering"],
                        "reception": processing_times["filtering"],
                        "formulation": processing_times["formulation"],
                        "packaging": processing_times["packaging"]
                    }

    for task_name in machine_requirements.keys():
        task = tasks_dict[task_name]
        machine_type = machine_requirements[task_name]
        duration = task_durations[task_name]
        for machine in machines_dict[machine_type]:
            model.add_mode(task, machine, duration)

    print(f'For a new product all modes are added')

    # Add temporal constraints
    # filtering starts after preparation
    model.add_end_before_start(tasks_dict["preparation"], tasks_dict["filtering"])

    # formulation starts after filtering
    model.add_end_before_start(tasks_dict["filtering"], tasks_dict["formulation"])

    # packaging starts after formulation
    model.add_end_before_start(tasks_dict["formulation"], tasks_dict["packaging"])

    # filtering and reception should happen at the same time
    model.add_start_before_start(tasks_dict["reception"], tasks_dict["filtering"])
    model.add_start_before_start(tasks_dict["filtering"], tasks_dict["reception"])
    model.add_end_before_end(tasks_dict["reception"], tasks_dict["filtering"])
    model.add_end_before_end(tasks_dict["filtering"], tasks_dict["reception"])

    print(f'A new product is added with all temporal constraints')

    # the formulation and the reception happen at the same tank
    model.add_identical_resources(tasks_dict["reception"], tasks_dict["formulation"])

# Add set-up times for the filters
for j, job in enumerate(jobs):
    for k, job in enumerate(jobs):
        task_j = job_tasks[j]["filtering"]
        task_k = job_tasks[k]["filtering"]
        set_up = random.randint(0, 3)
        for machine in machines_dict["plate_filter"]:
            model.add_setup_time(machine, task_j, task_k, set_up)
print(f'Set up times are added to the filter')

result = model.solve(display=False, time_limit=30)
print(result)

for task in result.best.tasks:
    print(task)

from pyjobshop.plot import plot_machine_gantt

plot_machine_gantt(result.best, model.data())
plt.savefig(f"pyjobshop_schedule_uflex_{nr_products}.png")

