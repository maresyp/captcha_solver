from typing import Optional
import cv2 as cv  # type: ignore
import numpy as np
from image_recognition.matcher import Matcher


class TemplateMatcher(Matcher):
    __MATCHING_METHOD = cv.TM_CCOEFF_NORMED

    # TODO: rewrite this
    def match_one(self, needle: np.ndarray, threshold: float) -> Optional[tuple[int, int]]:
        """ Try to find matching needle in haystack """

        _, max_val, _, max_loc = cv.minMaxLoc(
            cv.matchTemplate(self.haystack, needle, self.__MATCHING_METHOD)
        )
        return max_val, max_loc if max_val >= threshold else None

    def match_multiple(self, needle: np.ndarray, threshold: float, max_results=10) -> Optional[list[np.ndarray]]:
        """
        Search for needles in haystack \n
        returns array of rectangles around found objects
        :return: np.ndarray[np.ndarray[int, int, int, int]]
        """
        match_result: np.ndarray = cv.matchTemplate(
            self.haystack, needle, self.__MATCHING_METHOD)
        locations: list[tuple] = list(
            zip(*np.where(match_result >= threshold)[::-1]))
        if not locations:
            return None

        # rect -> [x, y, width, height]
        rectangles: list = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), needle.shape[1], needle.shape[0]]
            rectangles.append(rect)
            rectangles.append(rect)
        rectangles, _ = cv.groupRectangles(rectangles, 1, 0.5)
        return rectangles[:max_results] if len(rectangles) else None
