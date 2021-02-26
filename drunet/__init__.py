import cv2 as cv
from tf_package import utils

paths = utils.list_all_file(r'D:\Users\Ying\Desktop\DR-UNet\DR-UNet\figures')

for i, path in enumerate(paths):
    image = cv.imread(path)
    cv.imwrite('Fig{}.jpg'.format(i), image)
    cv.imwrite()