import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
from createFilterBank import create_filterbank
from skimage import color
import matplotlib.pyplot as plt
import cv2 as cv
import pickle


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    """
    H, W, _ = I.shape
    filterResponses = extract_filter_responses(I, filterBank)
    wordMap = np.zeros(H * W)
    for i in range(len(filterResponses)):
        distance = cdist(filterResponses[i, :], dictionary, 'euclidean')
        tag = np.min(distance)
        wordMap[i] = tag
    return wordMap.reshape((H, W))

    """
    res = extract_filter_responses(I, filterBank)
    cols, rows,_ = I.shape
    wordMap = np.zeros(cols * rows)
    n = len(filterBank)

    res = res.reshape((rows * cols, 3 * n))

    D = cdist(res, dictionary, 'euclidean')


    for i in range(rows * cols):
        wordMap[i] = np.argmin(D[i])


    return wordMap.reshape((cols, rows))


    # ----------------------------------------------

from skimage.color import hsv2rgb
def set_color_mapping(imL, K):
    '''
    This function converts wordmap into colormaps such that color for each cluster is bright and
    different, and doesn't depend on range of labels, unlike matplotlib or cmap of scipy

    [input]
    * imL: wordmap
    * K: total number of cluster-centers

    [output]
    * colormap: colormap for the wordmap
    '''

    im_h = 0.7 * imL / (1.0 * K) + 0.3 * (imL % (K // 10)) / 10.001
    im_s = 0.7 + 0.3 * ((imL) % (K // 20)) / (20.001)
    im_v = np.ones(imL.shape)  # 0.8 + 0.2*((K - imL) % (K//5)) / (5.01)
    imL_hsv = np.stack((im_h, im_s, im_v), axis=2)
    imL_rgb = hsv2rgb(imL_hsv)
    return (imL_rgb * 255.0).astype(np.uint8)

def test_get_visual_words():
    airport_imagenames = ['sun_aesovualhburmfhn.jpg', 'sun_aesyuxjawitlduic.jpg',
                          'sun_aetygbcukodnyxkl.jpg']
    desert_imagenames = ['sun_acqlitnnratfsrsk.jpg', 'sun_acztaebqvjaggqyh.jpg',
                         'sun_adpbjcrpyetqykvt.jpg']

    dictionaryHarris = pickle.load(open('dictionaryHarris.pkl', 'rb'))
    dictionaryRandom = pickle.load(open('dictionaryRandom.pkl', 'rb'))

    filterBank = create_filterbank()


    for i, path in enumerate(airport_imagenames):
        I = cv.imread('../data/airport/' + path)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)


        wordMapHarris = get_visual_words(I, dictionaryHarris, filterBank)
        wordMapRandom = get_visual_words(I, dictionaryRandom, filterBank)
        wordImageHarris = color.label2rgb(wordMapHarris, bg_label=0)
        wordImageRandom = color.label2rgb(wordMapRandom, bg_label=0)

        plt.imshow(wordImageHarris)

        plt.savefig('../results/2.1/airport.' + str(i) + 'Harris.jpg')
        breakpoint()

        plt.imshow(wordImageRandom)

        plt.savefig('../results/2.1/airport.' + str(i) + 'Random.jpg')
        breakpoint()


    for i, path in enumerate(desert_imagenames):
        I = cv.imread('../data/desert/' + path)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

        wordMapHarris = get_visual_words(I, dictionaryHarris, filterBank)
        wordMapRandom = get_visual_words(I, dictionaryRandom, filterBank)

        wordImageHarris = color.label2rgb(wordMapHarris, bg_label=0)
        wordImageRandom = color.label2rgb(wordMapRandom, bg_label=0)

        plt.imshow(wordImageHarris, cmap='hsv')

        plt.savefig('../results/2.1/desert.' + str(i) + 'Harris.jpg')
        breakpoint()

        plt.imshow(wordImageRandom, cmap='hsv')
        plt.savefig('../results/2.1/desert.' + str(i) + 'Random.jpg')
        breakpoint()
