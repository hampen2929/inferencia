from dataclasses import dataclass
import numpy as np


@dataclass
class ObjectTrackingResult:
    frame_index: int
    tracking_id: int
    class_id: int
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    confidence: float
    is_active: bool

    def to_txt(self):
        return "{} {} {} {} {} {}".format(self.class_name,
                                          self.confidence,
                                          self.xmin,
                                          self.ymin,
                                          self.xmax,
                                          self.ymax)

    def to_dict(self):
        return self.__dict__

    def to_array(self):
        return np.array([self.xmin,
                         self.ymin,
                         self.xmax,
                         self.ymax,
                         self.confidence,
                         ])

    def to_list(self):
        return [self.xmin,
                self.ymin,
                self.xmax,
                self.ymax,
                self.confidence,
                ]
