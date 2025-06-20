from src.entities.factoryData import FactoryData
from src.entities.JSONParsable import JSONParseable
import json


class Instance:
    def __init__(self, product_ids: list[int], due_dates: list[int], objective_weights: dict, factory: FactoryData):
        self.product_ids = product_ids
        self.due_dates = due_dates
        self.objective_weights = objective_weights
        self.factory = factory

    def to_dict(self):
        return {
            "product_ids": self.product_ids,
            "due_dates": self.due_dates,
            "objective_weights": self.objective_weights,
            "factory": self.factory.to_dict()
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            product_ids=data["product_ids"],
            due_dates=data["due_dates"],
            objective_weights=data["objective_weights"],
            factory=FactoryData.from_dict(data["factory"]))

