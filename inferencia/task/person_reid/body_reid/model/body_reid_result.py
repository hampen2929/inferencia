import numpy as np
from dataclasses import dataclass


@dataclass
class BodyReidResult:
    feature: np.ndarray

    def to_dict(self):
        return self.__dict__
