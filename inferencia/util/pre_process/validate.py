from typing import List, Union
import numpy as np


def validate_image(image: Union[np.ndarray, List]) -> List[np.ndarray]:
    """input image returned as List

    Args:
        image (Union[np.ndarray, List]): single image or list multi image

    Raises:
        ValueError: [description]

    Returns:
        List[np.ndarray]: [description]
    """
    if isinstance(image, np.ndarray):
        images_shape = image.shape
        if len(images_shape) == 3:
            "single image"
            images = [image]
        else:
            msg = "image shape length must be 3. Not {}.".format(
                len(images_shape))
            raise ValueError(msg)
    elif isinstance(image, list):
        images = image
    return images
