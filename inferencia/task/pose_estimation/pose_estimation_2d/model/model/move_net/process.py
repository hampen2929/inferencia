from typing import Tuple

import numpy as np


def change_ratio(image: np.ndarray,
                 height: int,
                 width: int,
                 ratio=16/9) -> Tuple[np.ndarray, int, int]:
    image_ratio = width / height
    if image_ratio < ratio:
        # imageが縦長
        canvas_height = height
        canvas_width = int(canvas_height * ratio)
        canvas = np.zeros(
            (canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[0:height, 0:width, :] = image
    elif image_ratio < ratio:
        # imageが横長
        canvas_width = width
        canvas_height = int(canvas_width / ratio)
        canvas = np.zeros(
            (canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[0:height, 0:width, :] = image
    else:
        canvas = image
    return canvas, canvas_height, canvas_width
