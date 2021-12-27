# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# February 21, 2020
# ##################################################################### #

import numpy as np
import os
import matplotlib.pyplot as plt

def renderBlinnPhongImage(brdfParams, normalIm, lightVec, viewVec):

    """
    Question 3


    Parameters
    ----------
    brdfParams : numpy.ndarray
        Triplet of Blinn-Phong parameters, size 1 x 3.

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    lightVec : numpy.ndarray
        Lighting vector of size 1 x 3

    viewVec : numpy.ndarray
        Viewing vector of size 1 x 3

    Returns
    -------
    image : numpy.ndarray
        The rendered image
    """

    kd, ks, a = brdfParams[0]
    h = (viewVec + lightVec) / np.linalg.norm(viewVec + lightVec)

    image = np.zeros(normalIm.shape[0:1])

    for i in range(len(normalIm)):
        n = normalIm[i].T

        image[i] = (kd * (n.dot(lightVec.T))) + (ks * pow((h.dot(n.T)), a))

    return image

if __name__ == '__main__':

    # Put your main code for Question 3 here
    # You can import functions for photometric stereo from Question 2


    bunny = np.load('../data/bunny.npy')
    save = os.path.isdir('../results')


    # 3.1
    # 45 degrees up
    h, w = bunny.shape[0:2]
    normalIm = np.reshape(bunny, (w * h, 3))
    lightVec = np.array([[0, np.sqrt(2) / 2, np.sqrt(2) / 2]])
    viewVec = np.array([[0, 0, 1]])
    brdfParams1 = np.array([[.3, .5, 1]])
    brdfParams2 = np.array([[.5, 0, 5]])
    brdfParams3 = np.array([[.3, .5, 20]])
    brdfParams4 = np.array([[.3, .5, 40]])

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams1, normalIm, lightVec, viewVec), (h,w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45up-params1.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams2, normalIm, lightVec, viewVec), (h,w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45up-params2.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams3, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45up-params3.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams4, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45up-params4.png")

    # 45 right
    lightVec = np.array([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])
    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams1, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45right-params1.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams2, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45right-params2.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams3, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45right-params3.png")

    rendered_bunny = np.reshape(renderBlinnPhongImage(brdfParams4, normalIm, lightVec, viewVec), (h, w))
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p3-1-45right-params4.png")

    #3.2
    h, w = bunny.shape[0:2]
    normalIm = np.reshape(bunny, (w * h, 3))
    brdfParams = np.array([[.3, .5, 10]])
    viewVec = np.array([[0, 0, 1]])

    lightVec1 = np.array([[.1, 0, .9]])
    rendered_bunny1 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec1, viewVec), (h, w))
    rendered_bunny1 = rendered_bunny1.clip(0,100000)

    lightVec2 = np.array([[.1, .1, .9]])
    rendered_bunny2 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec2, viewVec), (h, w))
    rendered_bunny2 = rendered_bunny2.clip(0,100000)

    lightVec3 = np.array([[-.1, -0.1, .9]])
    rendered_bunny3 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec3, viewVec), (h, w))
    rendered_bunny3 = rendered_bunny3.clip(0,100000)

    s = rendered_bunny1.shape
    h, w = s
    I = np.array([rendered_bunny1.reshape((h * w)),rendered_bunny2.reshape((h * w)),rendered_bunny3.reshape((h * w))])
    L = np.array([[.1, 0, .9],
                  [.1, .1, .9],
                  [-.1, -0.1, .9]]).T

    from q32_helper import estimatePseudonormalsCalibrated, estimateAlbedosNormals, displayAlbedosNormals
    from q32_helper import estimateShape,plotSurface

    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    surface = estimateShape(normals, s)
    plotSurface(surface)
    
    # 3.3
    h, w = bunny.shape[0:2]
    normalIm = np.reshape(bunny, (w * h, 3))
    brdfParams = np.array([[1, 0, 1]])
    viewVec = np.array([[0, 0, 1]])

    lightVec1 = np.array([[.1, 0, .9]])
    rendered_bunny1 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec1, viewVec), (h, w))
    rendered_bunny1 = rendered_bunny1.clip(0, 100000)

    lightVec2 = np.array([[.1, .1, .9]])
    rendered_bunny2 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec2, viewVec), (h, w))
    rendered_bunny2 = rendered_bunny2.clip(0, 100000)

    lightVec3 = np.array([[-.1, -0.1, .9]])
    rendered_bunny3 = np.reshape(renderBlinnPhongImage(brdfParams, normalIm, lightVec3, viewVec), (h, w))
    rendered_bunny3 = rendered_bunny3.clip(0, 100000)

    s = rendered_bunny1.shape
    h, w = s
    I = np.array([rendered_bunny1.reshape((h * w)), rendered_bunny2.reshape((h * w)), rendered_bunny3.reshape((h * w))])
    L = np.array([[.1, 0, .9],
                  [.1, .1, .9],
                  [-.1, -0.1, .9]]).T

    from q33_helper import estimatePseudonormalsCalibrated, estimateAlbedosNormals, displayAlbedosNormals
    from q33_helper import estimateShape, plotSurface

    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    surface = estimateShape(normals, s)
    plotSurface(surface)

