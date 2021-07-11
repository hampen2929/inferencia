from abc import ABCMeta, abstractmethod


class TmpVisualizer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def visualize():
        raise NotImplementedError()
