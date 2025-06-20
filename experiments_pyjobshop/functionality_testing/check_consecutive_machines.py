from pyjobshop import Model
import matplotlib.pyplot as plt
from pyjobshop import Model

for identical in [True, False]:
    model = Model()
    jobs = [model.add_job() for _ in range(1)]
    tasks = [[model.add_task(job=job) for _ in range(2)] for job in jobs]

    machines_trans = ["v01_1", "v01_1", "v01_1", "v01_1"]
    machines = [model.add_machine(machines_trans[i]) for i in range(4)]


    i =0
    duration = 10
    for job in range(len(jobs)):
        for task in tasks[job]:

            if i == 0:
                model.add_mode(task, [machines[0], machines[1]], duration)
                model.add_mode(task, [machines[2], machines[3]], duration)
            else:
                model.add_mode(task, [machines[0], machines[1]], duration)
                model.add_mode(task, [machines[2], machines[3]], duration)


    first = tasks[0][0]
    second = tasks[0][1]
    if identical:
        model.add_identical_resources(first, second)
    
    result = model.solve(display=False)
    print(f'We now solve the model with identical constraint set to {identical}')
    print(result)

    for task in result.best.tasks:
        print(task)

    from pyjobshop.plot import plot_machine_gantt

    plot_machine_gantt(result.best, model.data())
    plt.savefig("pyjobshop_check_consecutive.png")

