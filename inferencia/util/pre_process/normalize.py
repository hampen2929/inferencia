import numpy as np


def normalize(img: np.ndarray,
              norm_mean: np.ndarray = np.array(
                  [0.485, 0.456, 0.406]),
              norm_std: np.ndarray = np.array(
                  [0.229, 0.224, 0.225])
              ) -> np.ndarray:
    img_resize = (img - 255 * norm_mean) / (255 * norm_std)
    img_resize = img_resize.transpose(2, 0, 1)
    img_resize = img_resize.astype(np.float32)
    return img_resize
