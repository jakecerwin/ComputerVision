import pickle
from createFilterBank import create_filterbank
import cv2 as cv
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import numpy as np
import os

def idf_get_image_features(wordMap, dictionarySize):
    h = np.histogram(wordMap[:], bins=dictionarySize, range=(0, dictionarySize))[0]
    hN = h.astype(np.bool_).astype(np.float_)  # normalize
    return hN


meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
test_imagenames = meta['test_imagenames']

dictionaryHarris = pickle.load(open('dictionaryRandom.pkl', 'rb'))

filterBank = create_filterbank()

train_labels = meta['train_labels']

T = len(train_imagenames)
K = len(dictionaryHarris)

train_features = np.zeros((T, K))

dwd = np.zeros(K)
for i, path in enumerate(train_imagenames):
    print('-- processing %d/%d' % (i, len(train_imagenames)))
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    pathP = '../data/' + path.split('.')[0] + '_Random.pkl'
    if os.path.isfile(pathP):
        wordMap = pickle.load(open(pathP, 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryHarris, filterBank)

    dwd += get_image_features(wordMap, K)

idf = np.log((np.ones(K) * T) / dwd)

pickle.dump(idf, open('idf.pkl', 'wb'))
