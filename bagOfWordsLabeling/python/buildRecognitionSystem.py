import pickle
from createFilterBank import create_filterbank
import cv2 as cv
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
import numpy as np
import os

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
test_imagenames = meta['test_imagenames']

dictionaryRandom = pickle.load(open('dictionaryRandom.pkl', 'rb'))
dictionaryHarris = pickle.load(open('dictionaryHarris.pkl', 'rb'))

filterBank = create_filterbank()

train_labels = meta['train_labels']

T = len(train_imagenames)
K = len(dictionaryRandom)

train_features = np.zeros((T, K))


#Random
for i, path in enumerate(train_imagenames):
    print('-- processing %d/%d' % (i, len(train_imagenames)))
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    pathP = '../data/' + path.split('.')[0] + '_Random.pkl'
    if os.path.isfile(pathP):
        wordMap = pickle.load(open(pathP, 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryRandom, filterBank)


    hist = get_image_features(wordMap, K)
    train_features[i, :] = hist

pickle.dump([train_features, dictionaryRandom, filterBank, train_labels],
            open('visionRandom.pkl', 'wb'))

# Harris
for i, path in enumerate(train_imagenames):
    print('-- processing %d/%d' % (i, len(train_imagenames)))
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    pathP = '../data/' + path.split('.')[0] + '_Harris.pkl'
    if os.path.isfile(pathP):
        wordMap = pickle.load(open(pathP, 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryHarris, filterBank)

    hist = get_image_features(wordMap, K)
    train_features[i, :] = hist

pickle.dump([train_features, dictionaryHarris, filterBank, train_labels],
            open('visionHarris.pkl', 'wb'))