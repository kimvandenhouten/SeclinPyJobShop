import matplotlib.pyplot as plt
from pyjobshop import Model
import numpy as np
from src.entities.temporalRelation import ConstraintType
from src.entities.instance import Instance
import pyjobshop


class CPModel:
    """
    A class that creates a concrete PyJobShop model for based on a scheduling instance for the dsm-firmenich factory.

    Attributes:
       model (PyJobShop model): A PyJobShop model.
       instance (instance): The instance object that is used to construct the PyJobShop model.
       nr_products (int): The number of products that are present in this instance.
       machines (List[Machine]): The PyJobShop machines that are in the model.
       machines_dict (Dict[str, Machine]): A dictionary that maps each machine name to the machine object.
       jobs (List[Job]): The PyJobShop jobs that are in the model.
       job tasks (Dict[str, Dict[str,Task]): A dictionary that maps each job name to the tasks.
    """

    def __init__(self, instance: Instance):
        self.model = Model()
        self.instance = instance
        self.nr_products = len(instance.product_ids)

        self.factory = instance.factory
        self.machines = []
        self.machines_dict = {}

        self.jobs = []
        self.job_tasks = {}

        self.model.set_objective(**instance.objective_weights)
        self.add_machines()
        self.add_jobs()  # TODO: base it on the job keys

    def add_machines(self):
        """
        Receives the instance object and add all machine objects to the PyJobShop model.
        """
        # Add all the machines to the model

        for i, name in enumerate(self.factory.resource_names):
            if self.factory.capacity[i] == 1:
                machine = self.model.add_machine(name=name)
                self.machines_dict[name] = machine
                self.machines.append(machine)
            else:
                for j in range(self.factory.capacity[i]):
                    machine = self.model.add_machine(name=name)
                    self.machines_dict[name] = machine
                    self.machines.append(machine)

    def add_jobs(self):
        """
        Receives the instance object and add all job objects to the PyJobShop model.
        """

        # TODO: base the instance on the job keys
        self.jobs = [self.model.add_job(due_date=self.instance.due_dates[i], name=str(int(self.instance.product_ids[i])))
                     for i in range(self.nr_products)]

        for j, job in enumerate(self.jobs):
            print(f'We start with job {job.name}')
            job_data = self.factory.products[int(job.name)]
            tasks = []
            tasks_dict = {}

            # Add tasks with durations
            for task_data in job_data.tasks:
                task = self.model.add_task(job=job, name=task_data.name, fixed_duration=task_data.fixed_duration)
                tasks.append(task)
                tasks_dict[task_data.name] = task

            self.job_tasks[j] = tasks_dict

            # Add modes
            for task_data in job_data.tasks:
                task = tasks_dict[task_data.name]
                modes = task_data.modes
                print(f'We now add task_data {task_data.name} with modes {task_data.modes}')

                for mode in modes:
                    machine_mode = [self.machines_dict[machine] for machine in mode[1]] if isinstance(mode[1], list) \
                        else self.machines_dict[mode[1]]
                    self.model.add_mode(task, machine_mode, int(mode[0]))

            # Add product constraints
            for temp_relation in job_data.temporal_relations:
                task1 = tasks_dict[temp_relation.task1]
                task2 = tasks_dict[temp_relation.task2]
                delay = int(temp_relation.delay) if temp_relation.delay is not None else 0
                if temp_relation.type == ConstraintType.StartToStart:
                    self.model.add_start_before_start(task1, task2, delay)
                elif temp_relation.type == ConstraintType.StartToFinish:
                    self.model.add_start_before_end(task1, task2, delay)
                elif temp_relation.type == ConstraintType.FinishToStart:
                    self.model.add_end_before_start(task1, task2, delay)
                elif temp_relation.type == ConstraintType.FinishToFinish:
                    self.model.add_end_before_end(task1, task2, delay)
                else:
                    raise Exception("Unknown constraint type {}".format(temp_relation.type))

            for (task1_name, task_2_name) in job_data.identical_resources:
                task1 = tasks_dict[task1_name]
                task2 = tasks_dict[task_2_name]
                self.model.add_identical_resources(task1, task2)

    def add_set_up_times(self):
        # First we add the setup times based on the cleaning for FAM/MF/UF
        fixed_set_up = self.factory.constants["cleaning_time_contamination"]
        # Add set-up times for FAM fraction between each different products with a contamination constraint
        for j, job_1 in enumerate(self.jobs):
            job_1_data = self.factory.products[int(job_1.name)]
            nr_fracs_job_1 = job_1_data.constants["UF_fractions"]

            for k, job_2 in enumerate(self.jobs):
                job_2_data = self.factory.products[int(job_2.name)]
                nr_fracs_job_2 = job_2_data.constants["UF_fractions"]

                if [job_1_data.key, job_2_data.key] in self.factory.pairs_contamination:
                    # Compare types
                    for frac_1 in range(nr_fracs_job_1):
                        for frac_2 in range(nr_fracs_job_2):
                            task_j = self.job_tasks[j][f"FAM/MF_frac_{frac_1}"]
                            task_k = self.job_tasks[k][f"FAM/MF_frac_{frac_2}"]
                            for machine_name in ["FAM1", "FAM2", "FAM3", 'MF1/2']:
                                machine = self.machines_dict[machine_name]
                                self.model.add_setup_time(machine, task_j, task_k, fixed_set_up)
                                print(F'WE ADDED A SETUP TIME for {job_1_data.key}, {job_2_data.key} on {machine_name}')
                            task_l = self.job_tasks[j][f"UF_frac_{frac_1}"]
                            task_m = self.job_tasks[k][f"UF_frac_{frac_2}"]
                            for machine_name in ["UF 8_9", "UF 11_12", "UF 14_15", 'UF 7']:
                                machine = self.machines_dict[machine_name]
                                self.model.add_setup_time(machine, task_l, task_m, fixed_set_up)
                                print(F'WE ADDED A SETUP TIME for {job_1_data.key}, {job_2_data.key} on {machine_name}')

        # Then we add the setup times between all fermentation tasks that are executed at the same machine
        spread_between_fermentation = 3
        for j, job_1 in enumerate(self.jobs):
            job_1_data = self.factory.products[int(job_1.name)]

            for k, job_2 in enumerate(self.jobs):
                job_2_data = self.factory.products[int(job_2.name)]

                # Add this between post and pre
                task_j = self.job_tasks[j][f"fermentation_post"]
                task_k = self.job_tasks[k][f"fermentation_prep"]
                for machine_name in ["V100", "V140", "V200", "V218", "V42"]:
                    machine = self.machines_dict[machine_name]
                    self.model.add_setup_time(machine, task_j, task_k, spread_between_fermentation)
                    print(F'WE ADDED A SETUP TIME for {job_1_data.key}, {job_2_data.key} on {machine_name}')

    def solve(self, solver="cpoptimizer", display=False, time_limit=np.inf, plotting=True, print_result=True,
              print_sol=True, output_file="plot_new_model.png"):
        """
        Solves the PyJobShop model using the PyJobShop solve method.

        Args:
            solver (str): A str to specify the CP solver, either "cpoptimizer" or "ortools",
            display (Bool): Whether to display the progress of the CP solver.
            time_limit (float): A time-limit that is given to the solver.
            plotting (Bool): Whether to plot the output schedule as a Gannt chart.
            print_result (Bool): Whether to print the result of the solver.
            print_sol (Bool): Whether to print the solution.
            output_file (str): Where to save the Gannt chart if we plotted it.

        Returns:
            result.status: The status of the PyJobShop CP solver upon termination.
        """

        result = self.model.solve(solver=solver, display=display, time_limit=time_limit)
        if print_result:
            print(result)

        for task in result.best.tasks:
            if print_sol:
                print(task)

        if plotting:
            if result.status.value == pyjobshop.SolveStatus.FEASIBLE or result.status.value == pyjobshop.SolveStatus.OPTIMAL:
                from pyjobshop.plot import plot_machine_gantt

                plot_machine_gantt(result.best, self.model.data())
                plt.savefig(output_file)
                plt.close()

        return result