class TaskData:
    """
    A class that stores all data relevant for one task that is one step in the production of one product that is
    in the dsm-firmenich factory.

    Attributes:
       id (int): The unique id of the task.
       name (str): The name of the product.
       duration (int): The duration of the task.
       resource_modes (list[str|list[str]]) The list of resource modes that are possible for executing this task.
       fixed_duration (bool): Whether this task has a fixed duration.
    """
    def __init__(self, id: int, name: str, duration: int, resource_modes: list[str | list[str]],
                 fixed_duration: bool = True):
        self.id = id
        self.name = name
        self.duration = duration
        self.resource_modes = resource_modes
        self.fixed_duration = fixed_duration

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "duration": self.duration,
            "resource_modes": self.resource_modes,
            "fixed_duration": self.fixed_duration
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            name=data["name"],
            duration=data["duration"],
            resource_modes=data["resource_modes"],
            fixed_duration=data["fixed_duration"])
