# ##################################################################### #
# 16385: Computer Vision Homework 4
# Carnegie Mellon University
# Spring 2020
# ##################################################################### #

import numpy as np
import os
import matplotlib.pyplot as plt
from q1 import renderLambertianImage
from helper import integrateFrankot
import cv2
# import packages here

def loadData(path = "../data/"):

    """
    Question 2

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """
    I_list = list()
    for i in range(7):
        In = plt.imread(path + '/female_0' + str(i + 1) + '.tif')

        In_grey = In.sum(axis=2) / 3
        s = In_grey.shape
        I_list.append(In_grey.reshape((s[0] * s[1])))



    I = np.asarray(I_list)
    L = np.load('../data/sources.npy').T

    return I, L, s


def estimatePseudonormalsCalibrated(I, L):

    """
    Question 2

    In photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """
    B = np.linalg.inv(L @ L.T) @ L @ I

    return B


def estimateAlbedosNormals(B):

    '''
    Question 2

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)

    normals = B / albedos
    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):
    save = os.path.isdir('../results')
    """
    Question 2

    From the estimated pseudonormals, display the albedo and normal maps

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = albedos.reshape(s)
    if save:
        plt.imshow(albedoIm, cmap='gray')
        plt.savefig("../results/render_p2-2-albedo.png")

    normalIm = normals.T.reshape((s[0], s[1], 3))


    normalIm2 = (normalIm + 1) / 2
    if save:
        plt.imshow(normalIm2)
        plt.savefig("../results/render_p2-2-normal.png")
    return albedoIm, normalIm


def estimateShape(normals, s):
    save = os.path.isdir('../results')
    """
    Question 2

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    zx = - (normals[0] / normals[2]).reshape(s)
    zy = - (normals[1] / normals[2]).reshape(s)

    surface = integrateFrankot(zx, zy)
    if save:
        plt.imshow(surface, cmap='gray')
        plt.savefig("../results/render_p2-5-depth.png")

    return surface


def plotSurface(surface):
    save = os.path.isdir('../results')
    """
    Question 2 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """
    Z = surface
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    (X, Y) = np.meshgrid(range(0, Z.shape[1], 1), range(0, Z.shape[0], 1))  # create the X and Y matrices
    Zd = np.max(Z) - Z  # Normalize Z such that more depth means less value for plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Zd, cmap=cm.coolwarm)

    if save:
        plt.savefig("../results/render_p2-5-3D_surface.png")

    plt.show()
    return None


if __name__ == '__main__':
    save = os.path.isdir('../results')
    I, L, s = loadData('../data/PhotometricStereo')
    B = estimatePseudonormalsCalibrated(I, L)
    albedos, normals = estimateAlbedosNormals(B)

    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    lightVec1 = np.array([[.58, -.58, -.58]])
    lightVec2 = np.array([[-.58, -.58, -.58]])
    h,w = albedoIm.shape

    p4a = renderLambertianImage(albedoIm.reshape(w * h), normalIm.reshape((w * h, 3)), lightVec1).reshape((h, w))
    p4b = renderLambertianImage(albedoIm.reshape(w * h), normalIm.reshape((w * h, 3)), lightVec2).reshape((h, w))
    if save:
        plt.imshow(p4a, cmap='gray')
        plt.savefig("../results/render_p2-4a.png")

        plt.imshow(p4b, cmap='gray')
        plt.savefig("../results/render_p2-4b.png")



    surface = estimateShape(normals, s)
    plotSurface(surface)




