from image_recognition.template_matching import TemplateMatcher
from image_recognition.utils import read_image

if __name__ == '__main__':

    tm = TemplateMatcher(read_image('../test_files/easy/sample_multiple/haystack.jpg'))
    x = tm.match_multiple(read_image('../test_files/easy/sample_multiple/needle.jpg'), .5)
    print(type(x))
    for re in x:
        print(re)
        tm.draw_rectangle(re)
    tm.show()
