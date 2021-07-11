from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np


class TmpModel(metaclass=ABCMeta):

    @abstractmethod
    def inference(image: Union[np.ndarray, list]) -> list:
        raise NotImplementedError
