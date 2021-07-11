from abc import ABCMeta, abstractmethod
from typing import Union, List
import numpy as np

from .body_reid_result import BodyReidResult


class BodyReidModel(metaclass=ABCMeta):

    @abstractmethod
    def inference(image: Union[np.ndarray, list]) -> List[BodyReidResult]:
        raise NotImplementedError
