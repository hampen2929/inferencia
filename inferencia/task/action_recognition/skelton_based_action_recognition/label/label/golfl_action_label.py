from enum import Enum


class GolfSwingLabelEnum(Enum):
    other = 0
    swing = 1


class GolfSwingLabel():
    def __init__(self):
        pass

    def to_json(self):
        label_dict = {}
        for l in GolfSwingLabelEnum:
            label_dict[l.value] = l.name
        return label_dict
