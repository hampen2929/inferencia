from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np


class ObjectTrackingVisualizer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def visualize(image: np.ndarray,
                  object_tracking_results: List) -> np.ndarray:
        raise NotImplementedError()
