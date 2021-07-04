from abc import ABCMeta, abstractmethod

import numpy as np


class PoseEstimation2DVilualizer(metaclass=ABCMeta):
    def __init__(self,
                 body_edges):
        self.body_edges = body_edges

    @abstractmethod
    def visualize(self,
                  image: np.ndarray,
                  pose: np.ndarray):
        raise NotImplementedError()
