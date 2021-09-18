from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class FrameData:
    ret: bool
    frame: np.ndarray
    frame_height: int
    frame_width: int
    frame_index: int
    frame_index_str: str
    frame_path: Path

    def show(self):
        plt.imshow(self.frame[:, :, ::-1])
        plt.show()
