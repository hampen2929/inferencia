import numpy as np
from PIL import Image


def resize(img: np.ndarray,
           height: int,
           width: int) -> np.ndarray:
    """resize result is compartible to torchvision.transforms.Resize

    Args:
        img (np.ndarray): [description]
        height (int): [description]
        width (int): [description]

    Returns:
        np.ndarray: [description]
    """
    img_pil = Image.fromarray(img)
    img_resize = img_pil.resize((width, height),
                                resample=Image.BILINEAR)
    img_resize = np.asarray(img_resize)
    return img_resize
