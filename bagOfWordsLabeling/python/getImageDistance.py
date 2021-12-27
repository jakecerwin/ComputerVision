import numpy as np
from utils import chi2dist

def get_image_distance(hist1, hist2, method):

    if method == 'euclidean':
        dist = np.sum(np.abs(hist1 - hist2))
    else:
        dist = chi2dist(hist1, hist2)
    return dist