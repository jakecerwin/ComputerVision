import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 as cv


def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    random.seed(42)
    H, W, c = I.shape
    points = np.zeros((alpha, 2))
    for a in range(alpha):
        points[a, 0] = random.randint(0, H - 1)
        points[a, 1] = random.randint(0, W - 1)

    # ----------------------------------------------

    return points.astype(np.int_)

def test_get_random_points():
    I = cv.imread('../data/desert/sun_adpbjcrpyetqykvt.jpg')
    points = get_random_points(I, 20)


    plt.imshow(I)
    plt.scatter(points[:,1], points[:,0])
    plt.show()
    return None