import os
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
plt.ion()
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)

    plt.imshow(bw)

    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    plt.show()


    # find the rows using..RANSAC, counting, clustering, etc.
    centers = [((bbox[3] + bbox[1]) // 2, (bbox[2] + bbox[0]) // 2, bbox[2] - bbox[0], bbox[3] - bbox[1]) for bbox in bboxes]
    centers = sorted(centers, key=lambda p: p[1])

    rows, row = list(), list()
    row_height = centers[0][1]
    heights = [bbox[2] - bbox[0] for bbox in bboxes]
    margin = sum(heights) / len(heights)
    for center in centers:
        height = center[1]
        if height < row_height + margin: # only check because its sorted
            row.append(center)
        else:
            row = sorted(row, key=lambda p: p[0]) # sort by x
            rows.append(row)
            row_height = center[1]
            row = [center]

    row = sorted(row, key=lambda p: p[0])
    rows.append(row)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    data = []
    for row in rows:
        row_data = []
        for x, y, h, w in row:
            crop = bw[y - h // 2:y + h // 2, x - w // 2:x + w // 2]
            if h < w:
                crop = np.pad(crop, ((w // 8, w // 8), ((w - h) // 2 + w // 8, (w - h) // 2 + w // 8)),
                              'constant', constant_values=(1, 1))
            elif h > w:
                crop = np.pad(crop, ((h // 8, h // 8), ((h - w) // 2 + h // 8, (h - w) // 2 + h // 8)),
                              'constant', constant_values=(1, 1))

            crop = skimage.transform.resize(crop, (32, 32)).T
            crop = skimage.morphology.erosion(crop, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])) # added for better reseults

            row_data.append(crop.flatten())
        data.append(np.array(row_data))


    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))

    for row_data in data:
        h1 = forward(row_data, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        row = ''
        for i in range(probs.shape[0]):
            max = np.where(probs[i, :] == np.max(probs[i, :]))[0][0]

            if max < 26:
                row += chr(65 + max)
            else:
                row += chr(48 + max-26)

        print(row)