from abc import ABCMeta, abstractmethod
from typing import Union
import numpy as np


class SkeltonBasedActionRecognitionModel(metaclass=ABCMeta):

    @abstractmethod
    def pre_process():
        raise NotImplementedError

    @abstractmethod
    def post_process():
        raise NotImplementedError

    @abstractmethod
    def forward():
        raise NotImplementedError

    @abstractmethod
    def inference(image: Union[np.ndarray, list]) -> list:
        raise NotImplementedError
