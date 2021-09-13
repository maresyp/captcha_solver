from image_recognition.template_matching import TemplateMatcher
from image_recognition.utils import read_image

if __name__ == '__main__':

    matcher = TemplateMatcher(read_image(
        '../test_files/easy/sample_multiple/haystack.jpg'))
    x = matcher.match_multiple(read_image(
        '../test_files/easy/sample_multiple/needle.jpg'), .5)
    if x is not None:
        for re in x:
            print(re)
            matcher.draw_cross(matcher.get_center_point(re))
    matcher.show()
