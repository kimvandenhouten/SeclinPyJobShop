from enum import Enum


class ConstraintType(Enum):
    StartToStart = "start_to_start"
    StartToFinish = "start_to_finish"
    FinishToStart = "finish_to_start"
    FinishToFinish = "finish_to_finish"


class TemporalRelation:
    """
    A class that stores all data relevant for one Temporal Relation.

    Attributes:
       delay (float): Refers to the delay or lag of the temporal relation.
       type (ConstraintType): Refers to the type of temporal relation.
       task1 (str): Refers to the first task of the temporal relation.
       task2 (str): Refers to the second task of the temporal relation.
    """
    def __init__(self, task1, task2, delay: float = None, type: ConstraintType = None):
        self.delay = delay
        self.type = type
        self.task1 = task1
        self.task2 = task2

    def to_dict(self):
        return {
            "task1": self.task1,
            "task2": self.task2,
            "delay": self.delay,
            "type": self.type.value if self.type else None
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            task1=data["task1"],
            task2=data["task2"],
            delay=data["delay"],
            type=ConstraintType(data["type"])
            )
