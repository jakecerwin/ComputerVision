import pickle
import numpy as np
import cv2 as cv
from sklearn import svm
import os
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
test_imagenames = meta['test_imagenames']

trainFeatures, dictionaryRandom, filterBank, trainLabels = pickle.load(open('visionRandom.pkl', 'rb'))

testLabels = meta['test_labels']

TestSize = len(test_imagenames)
TrainSize = len(train_imagenames)

K = len(dictionaryRandom)


clf_Random_linear = svm.SVC(kernel='linear')
clf_Random_rbf = svm.SVC(kernel='rbf')
clf_Harris_rbf = svm.SVC(kernel='rbf')
clf_Harris_linear = svm.SVC(kernel='linear')

test_features = np.zeros((TestSize, K))


# Random
clf_Random_linear.fit(trainFeatures, trainLabels - 1)
clf_Random_rbf.fit(trainFeatures, trainLabels - 1)

for i, path in enumerate(test_imagenames):
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    if os.path.isfile('../data/' + path.split('.')[0] + '_Random.pkl'):
        wordMap = pickle.load(open('../data/' + path.split('.')[0] + '_Random.pkl', 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryRandom, filterBank)

    hist1 = get_image_features(wordMap, K)

    test_features[i, :] = hist1

pred_labels_random_linear = clf_Random_linear.predict(test_features)
pred_labels_random_rbf = clf_Random_rbf.predict(test_features)


# Harris
trainFeatures, dictionaryHarris, filterBank, trainLabels = pickle.load(open('visionHarris.pkl', 'rb'))

clf_Harris_linear.fit(trainFeatures, trainLabels - 1)
clf_Harris_rbf.fit(trainFeatures, trainLabels - 1)

for i, path in enumerate(test_imagenames):
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    if os.path.isfile('../data/' + path.split('.')[0] + '_Harris.pkl'):
        wordMap = pickle.load(open('../data/' + path.split('.')[0] + '_Harris.pkl', 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryHarris, filterBank)

    hist1 = get_image_features(wordMap, K)

    test_features[i, :] = hist1

pred_labels_harris_linear = clf_Harris_linear.predict(test_features)
pred_labels_harris_rbf = clf_Harris_rbf.predict(test_features)


hl, hr, rr, rl = 0,0,0,0

for i in range(TestSize):
    tl = testLabels[i] - 1

    if tl == pred_labels_harris_linear[i]:
        hl += 1
    if tl == pred_labels_harris_rbf[i]:
        hr += 1
    if tl == pred_labels_random_linear[i]:
        rl += 1
    if tl == pred_labels_random_rbf[i]:
        rr += 1
print("harris points with linear SVM: ")
print(hl / TestSize)
print("harris points with rbf SVM: ")
print(hr / TestSize)
print("random points with linear SVM: ")
print(rl / TestSize)
print("random points with rbf SVM: ")
print(rr / TestSize)