import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, localScatteringR, correlationNormalized_grid

def pilotAssignmentRandom(K, L, N, tau_p, APpositions, UEpositions, gainOverNoisedB, R, mode='random'):
    """return the pilot allocation
    INPUT>
    :param K: number of users
    :param L: number of APs
    :param N: number of antennas at the BS
    :param tau_p: Number of orthogonal pilots
    :param APpositions: matrix of dimensions Lx1 containing the APs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param UEpositions: matrix of dimensions Lx1 containing the UEs' locations as complex numbers,
                        where the real part is the horizontal position and the imaginary part is the
                        vertical position
    :param gainOverNoisedB: matrix with dimensions LxK where element (l,k) is the channel gain
                            (normalized by noise variance) between AP l and UE k
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    :param mode: select the pilot assignment mode
    OUTPUT>
    pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # to store pilot assignment
    pilotIndex = np.zeros((K), int)

    # random pilot assignment
    if mode == 'random':
        # random assignment
        pilotIndex = np.random.randint(0, tau_p, (K))

        # draw pilot assignment
        drawPilotAssignment(UEpositions, APpositions, pilotIndex, title='Random Pilot Assignment')

    return pilotIndex


def drawPilotAssignment(UEpositions, APpositions, pilotIndex, title):
    """
    INPUT>
    :param UEpositions: (see above)
    :param pilotIndex: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    OUTPUT>
    """

    # create a custom color palette for up to 10 orthogonal pilots
    custom_colors = np.array(['red', 'dodgerblue', 'green', 'orchid', 'aqua', 'orange', 'lime', 'black', 'pink', 'yellow']*10)

    # pilot assignment graph
    plt.scatter(UEpositions.real, UEpositions.imag, c=custom_colors[pilotIndex], marker='*')
    plt.scatter(APpositions.real, APpositions.imag, c='mediumblue', marker='^')
    plt.title(title)
    for i, txt in enumerate(range(len(UEpositions))):
        plt.annotate(txt, (UEpositions[i].real, UEpositions[i].imag))
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.show()

