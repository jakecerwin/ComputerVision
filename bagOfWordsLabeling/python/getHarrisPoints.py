import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter
import skimage
import matplotlib.pyplot as plt

class LocalMaxima(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value

def getPaddedValue(img, x, y):
    height = len(img)
    width = len(img[0])

    if 0 <= x < width and 0 <= y < height:
        return img[y][x]

    """ We consider 8 cases based on which sector non image coordinates fall
     1  |   2   |  3
    ____|_______|____
        |  THE  |   
     4  | IMAGE |  5
    ____|_______|____   
        |       |
      6 |   7   |  8
    """

    if x < 0 and y < 0:                 #Case 1
        return img[0][0]
    elif 0 <= x < width and y < 0:       #Case 2
        return img[0][x]
    elif width <= x and y < 0:          #Case 3
        return img[0][width-1]
    elif x < 0 and 0 <= y < height:      #Case 4
        return img[y][0]
    elif width <= x and 0 <= y < height: #Case 5
        return img[y][width-1]
    elif x < 0 and y < height:         #Case 6
        return img[height-1][0]
    elif 0 <= x < width and y < height: #Case 7
        return img[height-1][x]
    else:                               #Case 8
        return img[height-1][width-1]


def nonMaximumSuppresion(img, n):
    local_maxima = []
    min_maxima = 0
    height = len(img)
    width = len(img[0])
    for x in range(width):
        for y in range(height):
            local_max = img[y][x]
            if local_max < min_maxima:
                continue

            val = 0
            for inc in [8,7,6,5,4,3,2,1]:
                j = inc % 3 - 1
                i = int(np.floor(inc / 3)) - 1

                val = getPaddedValue(img,x + i, y + j)
                if val > local_max:
                    break
            if val > local_max:
                continue
            else:
                lm = LocalMaxima(x,y,local_max)
                if len(local_maxima) < n :
                    local_maxima.append(lm)
                    local_maxima.sort(key=lambda x: x.value)
                else:
                    local_maxima[0] = lm
                    local_maxima.sort(key=lambda x: x.value)
                    min_maxima = local_maxima[0].value

    return local_maxima

def get_harris_points(I, alpha, k):

    points = np.zeros((alpha, 2))

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------

    H, W = I.shape

    dy, dx = np.gradient(I)
    Ixx = dx ** 2
    Ixy = dy * dx
    Iyy = dy ** 2


    R = np.zeros((H,W))
    for y in range(0, H, 1):
        for x in range(0, W, 1):

            yl = y - 1 if  y - 1 >= 0 else 0
            yh = y + 2 if y + 1 < H else H
            xl = x - 1 if x - 1 >= 0 else 0
            xh = x + 2 if x + 1 < W else W

            sxx = Ixx[yl:yh, xl:xh].sum()
            sxy = Ixy[yl:yh, xl:xh].sum()
            syy = Iyy[yl:yh, xl:xh].sum()

            M = np.array([[sxx, sxy],
                          [sxy, syy]])
            det = (sxx * syy) - (sxy ** 2)
            ktr = k * ((sxx * syy) ** 2)

            R[y][x] = det - ktr


    local_maxima = nonMaximumSuppresion(R, alpha)

    for i in range(alpha):
        lm = local_maxima[i]
        points[i] = [lm.y ,lm.x]


    # ----------------------------------------------
    
    return points.astype(np.int_)

def test_get_harris_points():
    I = cv.imread('../data/campus/sun_abslhphpiejdjmpz.jpg')
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    points = get_harris_points(I, 500, 0.05)

    plt.imshow(I)
    plt.scatter(points[:,1], points[:,0], s=2, c='#ff0000')
    plt.savefig('../results/1.2/2.jpg')
    plt.show()

    I = cv.imread('../data/campus/sun_dafbfhztompasyyb.jpg')
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    points = get_harris_points(I, 500, 0.05)

    plt.imshow(I)
    plt.scatter(points[:,1], points[:,0], s=2, c='#ff0000')
    plt.savefig('../results/1.2/1.jpg')
    plt.show()

    I = cv.imread('../data/campus/sun_dehqtundqdrraedj.jpg')
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    points = get_harris_points(I, 500, 0.05)

    plt.imshow(I)
    plt.scatter(points[:,1], points[:,0], s=2, c='#ff0000')
    plt.savefig('../results/1.2/3.jpg')
    plt.show()