from typing import List, Tuple
import numpy as np
import cv2


def pre_process(images: List[np.ndarray],
                input_width: int,
                input_height: int) -> Tuple[np.ndarray,
                                            List[int, int, int]]:
    img_in = []
    for image in images:
        resized = cv2.resize(image,
                             (input_width, input_height),
                             interpolation=cv2.INTER_LINEAR)
        # img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = np.transpose(resized, (2, 0, 1)).astype(np.float32)
        # img /= 255.0
        img_in.append(img)
    img_in = np.array(img_in)
    return img_in
