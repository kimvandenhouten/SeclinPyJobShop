from src.entities.taskData import TaskData
from src.entities.temporalRelation import TemporalRelation


class ProductData:
    """
    A class that stores all data relevant for one product that is produced in the dsm-firmenich factory.

    Attributes:
       id (int): The unique id of the product.
       name (str): The name of the product.
       tasks (list[TaskData]): List of tasks that are needed to produce this product.
       constants (dict[str,int]): A dict of constants relevant for the factory scheduling.
       temporal_relations (list[TemporalRelation]): The list of temporal relations between the different tasks.
    """
    def __init__(self, id: int, name: str, tasks: list = None, constants: dict = None,
                 temporal_relations: list[TemporalRelation] = None,
                 identical_resources: list = None) -> None:
        self.id = id
        self.name = name
        self.tasks = tasks if tasks is not None else []
        self.constants = constants
        self.temporal_relations = temporal_relations if temporal_relations is not None else []
        self.identical_resources = identical_resources if identical_resources is not None else [] # TODO: currently not in use?

    def add_task(self, task: TaskData):
        """
        Receives a task object and adds it to the list of tasks.
        """
        self.tasks.append(task)

    def add_temporal_constraints(self, temporal_relation: TemporalRelation):
        """
        Receives a TemporalRelation object and adds it to the list of temporal relations.
        """
        self.temporal_relations.append(temporal_relation)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "tasks": [t.to_dict() for t in self.tasks],
            "constants": self.constants,
            "temporal_relations": [tr.to_dict() for tr in self.temporal_relations],
            "identical_resources": self.identical_resources
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            name=data["name"],
            tasks=[TaskData.from_dict(t) for t in data["tasks"]],
            constants=data["constants"],
            temporal_relations=[TemporalRelation.from_dict(tr) for tr in data["temporal_relations"]],
            identical_resources=data["identical_resources"]
        )
