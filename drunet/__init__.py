import cv2 as cv

image = cv.imread(r'D:\Users\Ying\Desktop\Fig1.tif')
cv.imwrite('fig1.jpg', image)