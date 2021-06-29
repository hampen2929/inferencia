from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectDetectionResult:
    class_id: int
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float

    def to_txt(self):
        return "{} {} {} {} {} {}".format(self.class_name,
                                          self.confidence,
                                          self.xmin,
                                          self.ymin,
                                          self.xmax,
                                          self.ymax)
