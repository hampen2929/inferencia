from enum import Enum
from inferencia.task.person_reid.body_reid.model.model.osnet.model.osnet import osnet_x0_25


class BodyReidModelName(Enum):
    osnet_x0_25 = "osnet_x0_25"
    osnet_x0_5 = "osnet_x0_5"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in BodyReidModelName}

    @staticmethod
    def names():
        return [l.name for l in BodyReidModelName]

    @staticmethod
    def values():
        return [l.value for l in BodyReidModelName]
