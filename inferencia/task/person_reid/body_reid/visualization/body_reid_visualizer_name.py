from enum import Enum


class BodyReidVisualizerName(Enum):
    body_reid_visualizer = "BodyReidVisualizer"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in BodyReidVisualizerName}

    @staticmethod
    def names():
        return [l.name for l in BodyReidVisualizerName]

    @staticmethod
    def values():
        return [l.value for l in BodyReidVisualizerName]
