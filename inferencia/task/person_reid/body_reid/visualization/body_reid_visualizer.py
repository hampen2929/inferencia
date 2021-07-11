from abc import ABCMeta, abstractmethod


class BodyReidVisualizer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def visualize():
        raise NotImplementedError()
