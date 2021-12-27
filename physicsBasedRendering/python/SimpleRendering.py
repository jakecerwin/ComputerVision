# ##################################################################### #
# 16385: Computer Vision Homework 4
# Carnegie Mellon University
# Spring 2020
# ##################################################################### #

import numpy as np
import matplotlib.pyplot as plt
import os
# import packages here

def renderLambertianImage(albedoIm, normalIm, lightVec):

    """
    Question 1


    Parameters
    ----------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    lightVec : numpy.ndarray
        Lighting vector of size 1 x 3

    Returns
    -------
    image : numpy.ndarray
        The rendered image
    """
    image = np.zeros(albedoIm.shape)
    for i in range(len(normalIm)):
        n = normalIm[i]
        image[i] = n @ lightVec.T

    image = np.multiply(albedoIm, image)

    return image


if __name__ == '__main__':
    save = os.path.isdir('../results')

    bunny = np.load('../data/bunny.npy')
    if save:
        plt.imshow(bunny)
        plt.savefig("../results/render_normals.png")

    h, w = bunny.shape[0:2]
    albedoIm = np.ones(h * w)
    normalIm = np.reshape(bunny, (w * h, 3))

    lightVec = np.array([[0,0,1]])

    rendered_bunny = np.reshape(renderLambertianImage(albedoIm, normalIm, lightVec), bunny.shape[0:2])
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p1-1.png")

    # light source rotated 45◦ up
    lightVec = np.array([[0, np.sqrt(2) / 2, np.sqrt(2) / 2]])
    rendered_bunny = np.reshape(renderLambertianImage(albedoIm, normalIm, lightVec), bunny.shape[0:2])
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p1-2i.png")

    # light source rotated 45◦ right
    lightVec = np.array([[np.sqrt(2) / 2, 0, np.sqrt(2) / 2]])
    rendered_bunny = np.reshape(renderLambertianImage(albedoIm, normalIm, lightVec), bunny.shape[0:2])
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p1-2ii.png")

    # light source rotated 75◦ right
    lightVec = np.array([[np.sin(np.deg2rad(75)), 0, np.cos(np.deg2rad(75))]])
    rendered_bunny = np.reshape(renderLambertianImage(albedoIm, normalIm, lightVec), bunny.shape[0:2])
    if save:
        plt.imshow(rendered_bunny, cmap='gray')
        plt.savefig("../results/render_p1-2iii.png")


