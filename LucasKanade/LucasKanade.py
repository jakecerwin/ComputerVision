import numpy as np
from scipy.interpolate import RectBivariateSpline


def LeastSquares(A, f):
    u = np.linalg.inv(A.T @ A) @ A.T @ f
    return u


def LucasKanade(It, It1, rect):
    # Input: 
    #   It: template image
    #   It1: Current image
    #   rect: Current position of the object
    #   (top left, bot right coordinates: x1, y1, x2, y2)
    # Output:
    #   p: movement vector dx, dy
    
    # set up the threshold
    threshold = 0.01875
    maxIters = 100
    p = np.zeros(2)          
    x1,y1,x2,y2 = rect

    # put your implementation here

    y, x = np.arange(It.shape[0]), np.arange(It.shape[1])

    interp_spline = RectBivariateSpline(y, x, It)
    interp_spline1 = RectBivariateSpline(y, x, It1)
    iterNum = 0

    x = np.arange(x1, x2 + 0.5)
    y = np.arange(y1, y2 + 0.5)
    X, Y = np.meshgrid(x, y)
    interp_It = interp_spline.ev(Y, X)

    while True:

        # transform points
        x = np.arange(x1 + p[0], x2 + p[0] + 0.5)
        y = np.arange(y1 + p[1], y2 + p[1] + 0.5)
        X, Y = np.meshgrid(x, y)

        # gradient for jacobian
        interp_It1 = interp_spline1.ev(Y, X)
        interp_gradientY = interp_spline.ev(Y, X, dx=1, dy=0).flatten()     # Mb remove
        interp_gradientX = interp_spline1.ev(Y, X, dx=0, dy=1).flatten()

        # get jacobian [dI(x)/du , dI(x)/dv]
        jacobian = np.zeros((len(interp_gradientX), 2))     # eq 8
        jacobian[:, 0] = interp_gradientX                   # eq 8
        jacobian[:, 1] = interp_gradientY                   # eq 8

        # T(x) - I(W(x;p))
        b = interp_It.flatten() - interp_It1.flatten()      # eq 9

        # Heterogenous solution dp = H^{-1} @ J^T @ b
        dp = LeastSquares(jacobian, b)                      # eq 6

        # p <- p + dp
        p += dp

        iterNum += 1
        if iterNum >= maxIters or np.sum(np.square(dp)) < threshold:  # eq 4
            break
    return p
