from abc import ABCMeta, abstractmethod

from .frame_data import FrameData


class BaseReader(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def read(self) -> FrameData:
        raise NotImplementedError()
