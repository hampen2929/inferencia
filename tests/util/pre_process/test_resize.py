import pytest

import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Resize

from inferencia.util.pre_process.resize import resize


TEST_DOG_IMG = './data/dog.jpg'


@pytest.mark.parametrize('height, width', [
    (256, 128),
    (100, 100),
])
def test_resize(height, width):
    img = cv2.imread(TEST_DOG_IMG)

    # torchvision
    trns_rsz = Resize((height, width))
    img_pil = Image.fromarray(img)
    trns_rsz_img = trns_rsz(img_pil)
    trns_rsz_img_arr = np.asarray(trns_rsz_img)

    # pil and cv2
    rsz_img = resize(img, height, width)

    np.testing.assert_allclose(trns_rsz_img_arr,
                               rsz_img,
                               rtol=1e-03,
                               atol=1e-05)
