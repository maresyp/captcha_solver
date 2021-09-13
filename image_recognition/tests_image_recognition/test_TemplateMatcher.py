
import unittest
from image_recognition.template_matching import TemplateMatcher


class TemplateMatcherTest(unittest.TestCase):

    def test_imread(self):
        self.assertRaises(FileNotFoundError,
                          TemplateMatcher.imread, filename='non_existing_file')

    def test_basic(self):
        matcher = TemplateMatcher(TemplateMatcher.imread(
            'image_recognition/tests_image_recognition/test_files/easy/sample_1/hay.jpg'))
        x = matcher.match_multiple(TemplateMatcher.imread(
            'image_recognition/tests_image_recognition/test_files/easy/sample_1/needle_1.jpg'), .5)
        if x is not None:
            for re in x:
                print(re)
                matcher.draw_cross(matcher.get_center_point(re))

        result = matcher.match_one(TemplateMatcher.imread(
            'image_recognition/tests_image_recognition/test_files/easy/sample_1/needle_1.jpg'), .7)
        print(type(result), result)
