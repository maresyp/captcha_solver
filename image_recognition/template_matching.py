import numpy as np
import cv2


class TemplateMatcher:
    __last_needle_size: tuple[int, int] = (0, 0)  # (width, height)

    def __init__(self, haystack: str):
        """ Load files """
        self.haystack = cv2.imread(haystack, cv2.IMREAD_UNCHANGED)
        if self.haystack is None:
            raise FileNotFoundError(f"File '{haystack}' does not exist")

    def __str__(self):
        return f'{self.__class__.__name__} for {self.haystack}'

    def match_one(self, needle: str, method=cv2.TM_CCOEFF_NORMED) -> tuple[float, tuple[int, int]]:
        """ Try to find matching needle in haystack using given method """
        needle = cv2.imread(needle, cv2.IMREAD_UNCHANGED)
        if needle is None:
            raise FileNotFoundError(f"File '{needle}' does not exist")

        self.__last_needle_size = (needle.shape[1], needle.shape[0])

        _, max_val, _, max_loc = cv2.minMaxLoc(
            cv2.matchTemplate(self.haystack, needle, method)
        )
        return max_val, max_loc

    def match_multiple(self, needle: str, method=cv2.TM_CCOEFF_NORMED) -> tuple[float, tuple[int, int]]:
        pass

    def draw_rectangle(self, max_loc: tuple[int, int], size: tuple[int, int] = None):
        """
        TODO: fix this docstring
        Draw rectangle of needle size(x, y) at given max_loc
        If size is not specified use last used size of needle
        """
        if size is None:
            size = self.__last_needle_size
        cpy = self.haystack.copy()
        cv2.rectangle(
            cpy,
            max_loc,
            (max_loc[0] + size[0], max_loc[1] + size[1]),
            (0, 255, 255),
            2
        )
        cv2.imshow('Match', cpy)
        cv2.waitKey()


if __name__ == '__main__':
    x = TemplateMatcher('../test_files/easy/sample_1/hay.jpg')
    print(x.match_one('../test_files/easy/sample_1/needle_2.jpg'))
    ret = x.match_one('../test_files/easy/sample_1/needle_1.jpg')
    print(ret)
    x.draw_rectangle(ret[1])
