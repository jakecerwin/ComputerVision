import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import *
import matplotlib.pyplot as plt

def extract_filter_responses(I, filterBank):

    I = I.astype(np.float64)
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------

    Il = rgb2lab(I)

    n = len(filterBank)

    H, W, c = I.shape

    filterResponses = np.zeros((H, W, 3 * n))

    for i in range(n):
        filter = filterBank[i]
        filterResponse1 = imfilter(Il[:,:,0], filter)
        filterResponse2 = imfilter(Il[:,:,0], filter)
        filterResponse3 = imfilter(Il[:,:,0], filter)
        filterResponses[:,:, 3 * i] = filterResponse1
        filterResponses[:,:,3 * i + 1] = filterResponse2
        filterResponses[:,:,3 * i + 2] = filterResponse3



    # ----------------------------------------------
    
    return filterResponses

def test_extract_filter_responses():
    I = cv.imread('../data/desert/sun_adpbjcrpyetqykvt.jpg')
    filterResponses = extract_filter_responses(I, create_filterbank())


    for i in range(len(filterResponses[0][0])):
        L = filterResponses[:,:,i]
        L_viz = (L - np.min(L)) / (np.max(L) - np.min(L))
        plt.imshow(L_viz, cmap='gray')
        plt.savefig('../results/filterResponse'+ str(i) + '.jpg')

    return None