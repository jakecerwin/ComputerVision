import numpy as np
import cv2 as cv
from createFilterBank import create_filterbank
from extractFilterResponses import extract_filter_responses
from getRandomPoints import get_random_points
from getHarrisPoints import get_harris_points
from sklearn.cluster import KMeans
import os


def get_dictionary(imgPaths, alpha, K, method):

    filterBank = create_filterbank()

    pixelResponses = np.zeros((alpha * len(imgPaths), 3 * len(filterBank)))

    for i, path in enumerate(imgPaths):
        print('-- processing %d/%d' % (i, len(imgPaths)))
        image = cv.imread('../data/%s' % path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)    # convert the image from bgr to rgb, OpenCV use BGR by default
        
        # -----fill in your implementation here --------
        pathP = '../data/' + path.split('.')[0] + '_PixelResponse_' + method + '.pkl'
        if os.path.isfile(pathP):
            pixelResponses[(i * alpha): ((i + 1) * alpha)] = pickle.load(open(pathP, 'rb'))
            continue


        filterResponses = extract_filter_responses(image, filterBank)
        if method == 'Random':
            indexes = get_random_points(image, alpha)
        else:
            indexes = get_harris_points(image, alpha, .05)


        xs = indexes[:, 1]
        ys = indexes[:, 0]
        pixelResponse = filterResponses[ys, xs]
        pixelResponses[(i * alpha): ((i + 1) * alpha)] = pixelResponse
        pickle.dump(pixelResponse, open(pathP, 'wb'))


    pickle.dump(pixelResponses, open(method + 'TrainPixelResponses.pkl', 'wb'))

        # ----------------------------------------------

    print("kmeans")
    dictionary = KMeans(n_clusters=K, random_state=0, n_jobs=3, verbose=2).fit(pixelResponses).cluster_centers_
    return dictionary


import _pickle as pickle
objects = []
with (open('../data/traintest.pkl', "rb")) as f:

    imgPaths = pickle.load(f)

imgPaths = imgPaths['train_imagenames']

alpha = 200
K = 500
method = 'Random'
dictionary = get_dictionary(imgPaths, alpha, K, method).astype(np.float32)

print("pickling")
pickle.dump(dictionary, open('dictionary' + str(method) + 'TestLarge.pkl', 'wb'))
