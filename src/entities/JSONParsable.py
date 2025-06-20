import json


class JSONParseable:
    """
    This originates from the repository with Deepali, can we somehow use this for using JSON?
    """
    @classmethod
    def parse_if_needed(cls, items):
        result = []
        for obj in items:
            if isinstance(obj, cls):
                result.append(obj)
            elif isinstance(obj, dict):
                result.append(cls(**obj))
            else:
                raise TypeError(f"Invalid type of data provided. Needed: {cls.__name__} or dict. "
                                f"Provided: {type(obj).__name__}")
        return result