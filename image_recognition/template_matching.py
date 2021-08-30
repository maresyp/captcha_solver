from typing import Optional
import cv2
import numpy as np


class TemplateMatcher:
    __MATCHING_METHOD = cv2.TM_CCOEFF_NORMED

    def __init__(self, haystack: np.ndarray):
        self.haystack = haystack

    def __str__(self):
        return f'{self.__class__.__name__} for {self.haystack}'

    def match_one(self, needle: np.ndarray, threshold: float) -> Optional[tuple[float, tuple[int, int]]]:
        """ Try to find matching needle in haystack """

        _, max_val, _, max_loc = cv2.minMaxLoc(
            cv2.matchTemplate(self.haystack, needle, self.__MATCHING_METHOD)
        )
        return max_val, max_loc if max_val >= threshold else None

    def match_multiple(self, needle: str, amount: int, threshold: float) -> \
            Optional[list[tuple[float, tuple[int, int]]]]:
        return None

    def draw_rectangle(self, loc: tuple[int, int], size: tuple[int, int]) -> None:
        """
        TODO: fix this docstring
        Draw rectangle of needle size(x, y) at given max_loc
        If size is not specified use last used size of needle
        """
        cpy = self.haystack.copy()
        cv2.rectangle(cpy, loc, (loc[0] + size[0], loc[1] + size[1]), (0, 255, 255), 2)
        cv2.imshow('Match', cpy)
        cv2.waitKey()


if __name__ == '__main__':
    from utils import read_image

    tm = TemplateMatcher(read_image('../test_files/hard/haystack.jpg', cv2.IMREAD_GRAYSCALE))
    x = tm.match_one(read_image('../test_files/hard/needle_1.jpg', cv2.IMREAD_GRAYSCALE), .1)
    tm.draw_rectangle(x[1], (50, 50))
