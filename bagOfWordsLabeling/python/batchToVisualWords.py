import multiprocessing
import pickle
import math
import cv2 as cv
from createFilterBank import create_filterbank
from getVisualWords import get_visual_words
import os


def batch_to_visual_words(num_cores):

    print('using %d threads for getting visual words' % num_cores)

    meta = pickle.load(open('../data/traintest.pkl', 'rb'))
    all_imagenames = meta['all_imagenames']

    #dictionary = pickle.load(open('dictionary%s.pkl' % point_method, 'rb'))

    dictionaryHarris = pickle.load(open('dictionaryHarris.pkl', 'rb'))
    dictionaryRandom = pickle.load(open('dictionaryRandom.pkl', 'rb'))

    filterBank = create_filterbank()

    for j in range(len(all_imagenames)):
        img_ind = j

        if img_ind < len(all_imagenames):
            img_name = all_imagenames[img_ind]

            print('converting %d-th image %s to visual words' % (img_ind, img_name))
            image = cv.imread('../data/%s' % img_name)
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)  # convert the image from bgr to rgb

            if not os.path.isfile('../data/%s_%s.pkl' % (img_name[:-4], 'Harris')):
                wordMapHarris = get_visual_words(image, dictionaryHarris, filterBank)
                pickle.dump(wordMapHarris, open('../data/%s_%s.pkl' % (img_name[:-4], 'Harris'), 'wb'))

            if not os.path.isfile('../data/%s_%s.pkl' % (img_name[:-4], 'Random')):
                wordMapRandom= get_visual_words(image, dictionaryRandom, filterBank)
                pickle.dump(wordMapRandom, open('../data/%s_%s.pkl' % (img_name[:-4], 'Random'), 'wb'))

    print('batch to visual words done!')


if __name__ == "__main__":

    batch_to_visual_words(num_cores=1)

