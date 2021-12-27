import pickle
import numpy as np
import cv2 as cv
import os
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
test_imagenames = meta['test_imagenames']

trainFeatures, dictionaryRandom, filterBank, trainLabels = pickle.load(open('visionRandom.pkl', 'rb'))


TestSize = len(test_imagenames)
TrainSize = len(train_imagenames)

K = len(dictionaryRandom)

randomDistanceChi = np.zeros(TrainSize)
randomDistanceEuc = np.zeros(TrainSize)
harrisDistanceChi = np.zeros(TrainSize)
harrisDistanceEuc = np.zeros(TrainSize)

randomCorrectChi = 0
randomCorrectEuc = 0
harrisCorrectChi = 0
harrisCorrectEuc = 0

randomConfusionMatrixChi = np.zeros((8, 8))
randomConfusionMatrixEuc = np.zeros((8, 8))
harrisConfusionMatrixChi = np.zeros((8, 8))
harrisConfusionMatrixEuc = np.zeros((8, 8))

testLabels = meta['test_labels']

# Random
for i, path in enumerate(test_imagenames):
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    if os.path.isfile('../data/' + path.split('.')[0] + '_Random.pkl'):
        wordMap = pickle.load(open('../data/' + path.split('.')[0] + '_Random.pkl', 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryRandom, filterBank)

    hist1 = get_image_features(wordMap, K)

    for j in range(TrainSize):
        hist2 = trainFeatures[j, :]
        randomDistanceChi[j] = get_image_distance(hist1, hist2, 'chi2')
        randomDistanceEuc[j] = get_image_distance(hist1, hist2, 'euclidean')

    chiLabel = int(trainLabels[np.argmin(randomDistanceChi)] - 1)
    eucLabel = int(trainLabels[np.argmin(randomDistanceEuc)] - 1)


    trueLabel = int(testLabels[i] - 1)

    if trueLabel == chiLabel:
        randomCorrectChi += 1
    if trueLabel == eucLabel:
        randomCorrectEuc += 1

    randomConfusionMatrixChi[trueLabel][chiLabel] += 1
    randomConfusionMatrixEuc[trueLabel][eucLabel] += 1

trainFeatures, dictionaryHarris, filterBank, trainLabels = pickle.load(open('visionHarris.pkl', 'rb'))
# Harris
for i, path in enumerate(test_imagenames):
    I = cv.imread('../data/%s' % path)
    I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

    if os.path.isfile('../data/' + path.split('.')[0] + '_Harris.pkl'):
        wordMap = pickle.load(open('../data/' + path.split('.')[0] + '_Harris.pkl', 'rb'))
    else:
        wordMap = get_visual_words(I, dictionaryHarris, filterBank)


    hist1 = get_image_features(wordMap, K)

    for j in range(TrainSize):
        hist2 = trainFeatures[j, :]
        harrisDistanceChi[j] = get_image_distance(hist1, hist2, 'chi2')
        harrisDistanceEuc[j] = get_image_distance(hist1, hist2, 'euclidean')

    chiLabel = int(trainLabels[np.argmin(harrisDistanceChi)] - 1)
    eucLabel = int(trainLabels[np.argmin(harrisDistanceEuc)] - 1)

    trueLabel = int(testLabels[i] - 1)

    if trueLabel == chiLabel:
        harrisCorrectChi += 1
    if trueLabel == eucLabel:
        harrisCorrectEuc += 1

    harrisConfusionMatrixChi[trueLabel][chiLabel] += 1
    harrisConfusionMatrixEuc[trueLabel][eucLabel] += 1

print("The accuracy of Random with chi2 dist is      : " + str(float(randomCorrectChi) / TestSize))
print("The accuracy of Random with Euclidean dist is : " + str(float(randomCorrectEuc) / TestSize))

print("The Confusion matrix for random with chi2 is      :")
print(randomConfusionMatrixChi)
print("The Confusion matrix for random with Euclidean is :")
print(randomConfusionMatrixEuc)


print("The accuracy of Harris with chi2 dist is      : " + str(float(harrisCorrectChi) / TestSize))
print("The accuracy of Harris with Euclidean dist is : " + str(float(harrisCorrectEuc) / TestSize))

print("The Confusion matrix for harris with chi2 is      :")
print(harrisConfusionMatrixChi)
print("The Confusion matrix for harris with Euclidean is :")
print(harrisConfusionMatrixEuc)
