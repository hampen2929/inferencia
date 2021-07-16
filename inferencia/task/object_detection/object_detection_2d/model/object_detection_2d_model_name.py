from enum import Enum


class ObjectDetection2DModelName(Enum):
    yolo_v4 = "YoloV4"
    yolo_v4_middle = "YoloV4Middle"
    tiny_yolo_v4 = "TinyYoloV4"

    @staticmethod
    def to_dict():
        return {l.name: l.value for l in ObjectDetection2DModelName}

    @staticmethod
    def names():
        return [l.name for l in ObjectDetection2DModelName]

    @staticmethod
    def values():
        return [l.value for l in ObjectDetection2DModelName]
