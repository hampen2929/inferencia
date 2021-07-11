import pytest

import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import ToTensor, Normalize

from inferencia.util.pre_process.normalize import normalize


TEST_DOG_IMG = './data/dog.jpg'


def test_normalize():
    img = cv2.imread(TEST_DOG_IMG)

    # torchvision
    normalize_trns = Normalize(mean=np.array([0.485, 0.456, 0.406]),
                               std=np.array([0.229, 0.224, 0.225]))
    to_tensor = ToTensor()

    img_pil = Image.fromarray(img)
    tnsr = to_tensor(img_pil)
    trns_norm_img = normalize_trns(tnsr)
    trns_norm_img_arr = np.asarray(trns_norm_img)

    # pil and cv2
    norm_arr = normalize(img)

    np.testing.assert_allclose(trns_norm_img_arr,
                               norm_arr,
                               rtol=1e-03,
                               atol=1e-05)
