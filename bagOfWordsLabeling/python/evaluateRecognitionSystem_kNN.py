import pickle
import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
from getVisualWords import get_visual_words
from getImageFeatures import get_image_features
from getImageDistance import get_image_distance

meta = pickle.load(open('../data/traintest.pkl', 'rb'))

train_imagenames = meta['train_imagenames']
test_imagenames = meta['test_imagenames']


TestSize = len(test_imagenames)
TrainSize = len(train_imagenames)




distance = np.zeros(TrainSize)

correct = np.zeros(40)

confusion = np.zeros((8, 8))

testLabels = meta['test_labels']

accuracy = np.zeros(40)

classes = np.zeros((TestSize, 40))

trainFeatures, dictionaryRandom, filterBank, trainLabels = pickle.load(open('visionRandom.pkl', 'rb'))
K = len(dictionaryRandom)
for k in range(40):
    print('running for-- kNN : k = %d' % k)

    # Random Euc
    for i, path in enumerate(test_imagenames):
        #print('-- processing %d/%d' % (i, len(test_imagenames)))
        I = cv.imread('../data/%s' % path)
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)

        votes = np.zeros(8)

        if os.path.isfile('../data/' + path.split('.')[0] + '_Random.pkl'):
            wordMap = pickle.load(open('../data/' + path.split('.')[0] + '_Random.pkl', 'rb'))
        else:
            wordMap = get_visual_words(I, dictionaryRandom, filterBank)

        hist1 = get_image_features(wordMap, K)

        for j in range(TrainSize):
            hist2 = trainFeatures[j, :]
            distance[j] = get_image_distance(hist1, hist2, 'euclidean')

        sorted = np.argsort(distance)
        knearest = sorted[:k + 1]

        for n in range(k+1):
            label = int(trainLabels[knearest[n]] - 1)
            votes[label] += 1

        guessLabel = np.argmax(votes)
        classes[i][k] = guessLabel

        trueLabel = int(testLabels[i] - 1)

        if trueLabel == guessLabel:
            correct[k] += 1


bestk = np.argmax(correct) #returns earliest occurence
accuracy = correct / TestSize

for i in range(TestSize):
    real = int(testLabels[i] - 1)
    j = int(classes[i][bestk])
    confusion[real][j] += 1

print(confusion)
print(accuracy)

plt.plot(accuracy)
plt.show()
