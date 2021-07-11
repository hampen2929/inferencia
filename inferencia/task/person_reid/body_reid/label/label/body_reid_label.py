from enum import Enum


class BodyReidLabel(Enum):
    body_reid = 0

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in BodyReidLabel}

    @staticmethod
    def names():
        return [l.name for l in BodyReidLabel]

    @staticmethod
    def values():
        return [l.value for l in BodyReidLabel]
