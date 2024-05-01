import numpy as np
import cv2 as cv

imgl = cv.imread('sample/00L.jpg', cv.IMREAD_GRAYSCALE)
imgr = cv.imread('sample/00R.jpg', cv.IMREAD_GRAYSCALE)

stereo = cv.StreoBM_create(numDisparities=60, blockSize=21)
diff = stereo.compute(imgl, imgr)
depth = cv.convertScaleAbs(diff, alpha=255/diff.max())

cv.imshow('Left', imgl)
cv.waitKey(0)
cv.imshow('Depth', depth)
cv.waitKey(0)
cv.destroyAllWindows()