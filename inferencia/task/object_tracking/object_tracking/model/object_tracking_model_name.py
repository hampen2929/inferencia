from enum import Enum


class ObjectTrackingModelName(Enum):
    fastmot = "FastMOT"
    # sort = "SORT"
    # deep_sort = "DeepSORT"
    # fair_mot = "FairMOT"
    # jde = "JDE"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in ObjectTrackingModelName}

    @staticmethod
    def names():
        return [l.name for l in ObjectTrackingModelName]

    @staticmethod
    def values():
        return [l.value for l in ObjectTrackingModelName]
