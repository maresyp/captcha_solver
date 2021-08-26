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
