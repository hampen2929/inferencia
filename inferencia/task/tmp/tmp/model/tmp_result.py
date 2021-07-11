from dataclasses import dataclass


@dataclass
class TmpResult:
    tmp: int

    def to_dict(self):
        return self.__dict__
