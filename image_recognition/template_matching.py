from typing import Optional
import cv2 as cv
import numpy as np
from matcher import Matcher


class TemplateMatcher(Matcher):
    __MATCHING_METHOD = cv.TM_CCOEFF_NORMED

    def match_one(self, needle: np.ndarray, threshold: float) -> Optional[tuple[float, tuple[int, int]]]:
        """ Try to find matching needle in haystack """

        _, max_val, _, max_loc = cv.minMaxLoc(
            cv.matchTemplate(self.haystack, needle, self.__MATCHING_METHOD)
        )
        return max_val, max_loc if max_val >= threshold else None

    def match_multiple(self, needle: np.ndarray, threshold: float) -> Optional[np.ndarray]:
        """
        Search for needles in haystack \n
        returns array of rectangles around found objects
        :return: np.ndarray[np.ndarray[int, int, int, int]]
        """
        match_result: np.ndarray = cv.matchTemplate(self.haystack, needle, self.__MATCHING_METHOD)
        locations: list[tuple] = list(zip(*np.where(match_result >= threshold)[::-1]))
        # rect -> [x, y, width, height]
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), needle.shape[1], needle.shape[0]]
            rectangles.append(rect)
            rectangles.append(rect)
        rectangles, _ = cv.groupRectangles(rectangles, 1, 0.5)
        return rectangles if len(rectangles) else None


if __name__ == '__main__':
    from utils import read_image

    tm = TemplateMatcher(read_image('../test_files/easy/sample_multiple/haystack.jpg'))
    x = tm.match_multiple(read_image('../test_files/easy/sample_multiple/needle.jpg'), .5)
    print(type(x))
    for re in x:
        print(re)
        tm.draw_rectangle(re)
    tm.show()
# tm.draw_rectangle(x[1], (50, 50))
