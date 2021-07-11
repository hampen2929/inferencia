from enum import Enum


class GolfSwingLabel(Enum):
    other = 0
    swing = 1

    @staticmethod
    def to_json():
        label_dict = {}
        for l in GolfSwingLabel:
            label_dict[l.value] = l.name
        return label_dict
