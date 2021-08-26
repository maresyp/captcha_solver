import numpy as np
import cv2


class TemplateMatcher:
    def __init__(self, haystack: str, needle: str):
        """ Load files """
        self.haystack = cv2.imread(haystack, cv2.IMREAD_UNCHANGED)
        if self.haystack is None:
            raise FileNotFoundError(f"File '{haystack}' does not exist")
        self.needle = cv2.imread(needle, cv2.IMREAD_UNCHANGED)
        if self.needle is None:
            raise FileNotFoundError(f"File '{needle}' does not exist")

    def __str__(self):
        return f'{self.__class__.__name__} for {self.needle} in {self.haystack}'

    def match(self, method=cv2.TM_CCOEFF_NORMED) -> tuple[float, tuple[int, int]]:
        """ Try to find matching needle in haystack """
        _, max_val, _, max_loc = cv2.minMaxLoc(
            cv2.matchTemplate(self.haystack, self.needle, method)
        )
        return max_val, max_loc

    def draw_rectangle(self, max_loc: tuple[int, int]):
        """ Draw rectangle of needle size at given max_loc"""
        width = self.needle.shape[1]
        height = self.needle.shape[0]
        cv2.rectangle(
            self.haystack,
            max_loc,
            (max_loc[0] + width, max_loc[1] + height),
            (0, 255, 255),
            2
        )


if __name__ == '__main__':
    TemplateMatcher('', '').match()
