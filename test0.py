import numpy as np
import cv2 as cv
import time

v = '04'
x = 'jpg'
imgl = cv.imread(f'sample/{v}L.{x}', cv.IMREAD_GRAYSCALE)
imgr = cv.imread(f'sample/{v}R.{x}', cv.IMREAD_GRAYSCALE)
imgl = cv.resize(imgl, (0,0), fx=0.25, fy=0.25)
imgr = cv.resize(imgr, (0,0), fx=0.25, fy=0.25)
o = 40

for i in range(50):#4,5):
    print(i)

    stereo = cv.StereoBM.create(numDisparities=16*i, blockSize=21)
    diff = stereo.compute(imgl, imgr)
    depth = cv.convertScaleAbs(diff, alpha=255/diff.max())

    print(depth[len(depth)>>1])
    x = (len(depth[0])>>1)+o
    for y in range(len(depth)):
        depth[y][x] = 0
    
    cv.imshow('Depth', depth)
    if cv.waitKey(0) == 27:
        break

cv.destroyAllWindows()