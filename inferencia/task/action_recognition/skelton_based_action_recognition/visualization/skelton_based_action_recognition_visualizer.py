from abc import ABCMeta, abstractmethod
from typing import List
import numpy as np


class SkeltonBasedActionRecognitionVisualizer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def visualize(image: np.ndarray,
                  object_detection_results: List) -> np.ndarray:
        raise NotImplementedError()
