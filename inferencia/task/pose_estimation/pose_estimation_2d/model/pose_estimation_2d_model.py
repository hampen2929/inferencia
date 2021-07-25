from abc import ABCMeta, abstractmethod
from typing import List, Union
import numpy as np


class PoseEstimation2dModel(metaclass=ABCMeta):

    @abstractmethod
    def pre_process(self, images: Union[np.ndarray, List]) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def post_process(self, forwarded: List) -> List:
        raise NotImplementedError()

    @abstractmethod
    def forward(self, pre_processed:  np.ndarray) -> List:
        raise NotImplementedError()

    @abstractmethod
    def inference(self, images: Union[np.ndarray, List]) -> List:
        # pre_processed = self.pre_process(images)
        # forwarded = self.forward(pre_processed)
        # post_processed = self.post_process(forwarded)
        return NotImplementedError()
