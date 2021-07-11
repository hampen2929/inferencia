from enum import Enum


class TmpModelName(Enum):
    tmp = "Tmp"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in TmpModelName}

    @staticmethod
    def names():
        return [l.name for l in TmpModelName]

    @staticmethod
    def values():
        return [l.value for l in TmpModelName]
