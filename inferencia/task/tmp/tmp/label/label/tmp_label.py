from enum import Enum


class TmpLabel(Enum):
    tmp = 0

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in TmpLabel}

    @staticmethod
    def names():
        return [l.name for l in TmpLabel]

    @staticmethod
    def values():
        return [l.value for l in TmpLabel]
