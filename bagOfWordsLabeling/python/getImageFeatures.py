import numpy as np


def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------

    #h = np.zeros(dictionarySize)
    #words = wordMap[:]
    #breakpoint()

    #for i in range(dictionarySize):
    #    h[i] = np.sum(words == i)

    #h = h / np.linalg.norm(h)


    h = np.histogram(wordMap[:], bins=dictionarySize, range=(0,dictionarySize))[0]
    hN = h / np.sum(h) #normalize



    # ----------------------------------------------
    
    return hN
