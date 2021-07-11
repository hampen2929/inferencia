from enum import Enum


class TmpVisualizerName(Enum):
    tmp_visualizer = "TmpVisualizer"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in TmpVisualizerName}

    @staticmethod
    def names():
        return [l.name for l in TmpVisualizerName]

    @staticmethod
    def values():
        return [l.value for l in TmpVisualizerName]
