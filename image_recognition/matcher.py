import cv2 as cv
import numpy as np


class Matcher:
    """ Base class for Matchers """

    def __init__(self, haystack: np.ndarray):
        self.haystack = haystack

    def __str__(self):
        return f'{self.__class__.__name__} for {self.haystack}'

    def draw_rectangle(self, rectangle: list[int, int, int, int]) -> None:
        """ Draw rectangle """
        x, y, width, height = rectangle
        cv.rectangle(self.haystack, (x, y), (x + width, y + height), (0, 255, 0), 2)

    def show(self) -> None:
        """ Show haystack and wait for key """
        cv.imshow('Matches', self.haystack)
        cv.waitKey()
