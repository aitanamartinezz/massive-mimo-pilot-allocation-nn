import numpy as np
import numpy.linalg as alg
import sympy as sp
import scipy.linalg as spalg
import matplotlib.pyplot as plt
import random

def functionComputeNMSE_uplink(tau_p, N, K, L, R, pilotIndex):
    """ Compute the NMSE for evey user according to equation (4.21).
    Pilot transmitting power is assumed equal for all the UEs

    INPUT>
    :param tau_p: length of pilot sequences
    :param N: number of antennas per AP
    :param K: number of UEs
    :param L: number of APs
    :param R: matrix with dimensions N x N x L x K where (:,:,l,k) is the spatial correlation
                            matrix between  AP l and UE k (normalized by noise variance)
    :param pilotIndex: matrix with dimensions Kx1 containing the integer indexes of the pilots
                            assigned to the UEs


    OUTPUT>
    system_NMSE: sum of the NMSE values of all UEs
    UEs_NMSE: store NMSE values of all UEs
    average_NMSE: average NMSE per UE
    """

    # uplink pilot power in mW (same for all UEs)
    p = 100

    # vector of the BS assigned to each UEs. We will work with one BS assigned to all the UEs
    D = np.ones((1, K))

    # Store the N x N identity matrix
    eyeN = np.identity(N)

    # To store the interference (pilot sharing users' correlation matrices and noise) matrices according to Eq. 4.6.
    Psi = np.zeros((N, N, L, tau_p), dtype=complex)
    # To store the NMSE per UE
    UEs_NMSE = np.zeros(K)

    # Compute the Psi matrices
    # run over all the pilots
    for t in range(tau_p):
        # get the UEs that share pilot t
        pilotSharing_UEs, = np.where(pilotIndex == t)
        # run over all the APs
        for l in range(L):
            # use Eq.4.6 to
            Psi[:, :, l, t] = alg.inv(eyeN + sum([tau_p*p*R[:, :, l, k] for k in pilotSharing_UEs]))

    # Compute the UEs' NMSE values
    for k in range(K):

        if sum(D[:, k]) > 0:
            # get the pilot assigned to UE k
            t_k = pilotIndex[k]

            # get the APs serving user k
            serving_APs, = np.where(D[:, k] == 1)

            UEs_NMSE[k] = 1 - (sum([tau_p*p*np.trace(R[:, :, l, k]@Psi[:, :, l, t_k]@R[:, :, l, k]) for l in serving_APs])/
                                sum([np.trace(R[:, :, l, k]) for l in serving_APs])).real
        else:
            UEs_NMSE[k] = 0

    # sum of NMSE of all UEs
    system_NMSE = np.sum(UEs_NMSE)

    return system_NMSE, UEs_NMSE