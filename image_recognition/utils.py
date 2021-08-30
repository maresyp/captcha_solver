import cv2
import numpy as np


def read_image(filename: str, method=cv2.IMREAD_UNCHANGED) -> np.ndarray:
    """ Wrapper for cv2.imread() that raises exception when file is not loaded properly """
    img = cv2.imread(filename, method)
    if img is None:
        raise FileNotFoundError(f'{filename} not found')
    return img
