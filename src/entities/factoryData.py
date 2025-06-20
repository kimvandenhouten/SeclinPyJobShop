from src.entities.productData import ProductData


class FactoryData:
    """
    A class that stores all factory data relevant for the dsm-firmenich factory.

    Attributes:
       name (str): The name of the factory.
       resource_names (list[str]): The list of resources present in the factory.
       capcity (list[int]): The capacity per resource.
       constants (dict[str,int]): A dict of constants relevant for the factory scheduling.
    """

    def __init__(self, name: str, resource_names: list[str], capacity: list[int], constants: dict = None, products:
    list[ProductData] = [], pairs_contamination: list = None):
        self.name = name
        self.resource_names = resource_names
        self.capacity = capacity
        self.constants = constants
        self.products = products
        self.pairs_contamination = pairs_contamination

    def add_product(self, product: ProductData):
        """
        Receives a productData object and adds it to the list of tasks.
        """
        self.products.append(product)

    def to_dict(self):
        return {
            "name": self.name,
            "resource_names": self.resource_names,
            "capacity": self.capacity,
            "constants": self.constants,
            "products": [p.to_dict() for p in self.products],
            "pairs_contamination": self.pairs_contamination
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            name=data["name"],
            resource_names=data["resource_names"],
            capacity=data["capacity"],
            constants=data["constants"],
            products=[ProductData.from_dict(p) for p in data["products"]],
            pairs_contamination=data["pairs_contamination"]
        )
