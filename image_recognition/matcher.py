import cv2 as cv  # type: ignore
import numpy as np


class Matcher:
    """ Base class for Matchers """

    def __init__(self, haystack: np.ndarray):
        self.haystack = haystack

    def __str__(self):
        return f'{self.__class__.__name__} for {self.haystack}'

    def draw_rectangle(self, rectangle: np.ndarray, color: tuple[int, int, int] = (0, 255, 0)):
        """ Draw rectangle """
        x, y, width, height = rectangle
        cv.rectangle(self.haystack, (x, y), (x + width, y + height), color, 2)

    @staticmethod
    def get_center_point(rectangle: np.ndarray) -> tuple[int, int]:
        """ Calculate center point of rectangle """
        x, y, width, height = rectangle
        return x + width // 2, y + height // 2

    def draw_cross(self, point: tuple[int, int], color=(0, 255, 0)):
        """ Draw x at center point of rectangle """
        cv.drawMarker(
            self.haystack, (point[0], point[1]), color, cv.MARKER_CROSS)

    def show(self) -> None:
        """ Show haystack and wait for key """
        cv.imshow('Matches', self.haystack)
        cv.waitKey()
        
    @staticmethod
    def imread(filename: str, method=cv.IMREAD_UNCHANGED) -> np.ndarray:
        """ Wrapper for cv2.imread() that raises exception when file is not loaded properly """
        img = cv.imread(filename, method)
        if img is None:
            raise FileNotFoundError(f'{filename} not found')
        return img
