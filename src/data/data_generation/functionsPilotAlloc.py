import numpy as np
import itertools
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random
import math
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from functionsUtils import db2pow, localScatteringR, correlationNormalized_grid

def pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, gainOverNoisedB, R):
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
    best_pilot_index: vector whose entry pilotIndex[k] contains the index of pilot assigned to UE k
    """

    # esto nos genera todas las combinaciones posibles
    pilot_allocations = itertools.product(range(tau_p), repeat = K)

    # Inicializamos variables
    best_system_NMSE = np.inf
    best_pilot_index = None

    # Para cada combinación entre todas las combinaciones posibles
    for pilot in pilot_allocations:
    # Evaluamos cual es la mejor buscando aquella que minimice el NMSE
        pilotIndex = np.array(pilot)
        #print("Current pilot assignment:", pilotIndex)
        system_NMSE, _ = functionComputeNMSE_uplink(tau_p, N, K, L, R, pilotIndex)
        #print(system_NMSE)
        if system_NMSE < best_system_NMSE:
            # Actual#iza la mejor asignación de pilotos y el mejor NMSE del sistema
            best_pilot_index = pilotIndex
            best_system_NMSE = system_NMSE

    #print("El mejor es",best_pilot_index, 'con un NMSE DE', best_system_NMSE )
    # draw pilot assignment
    drawPilotAssignment(UEpositions, APpositions, best_pilot_index, title='Pilot Assignment')

    return best_pilot_index

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

