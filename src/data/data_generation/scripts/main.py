import numpy as np
import csv
from functionsSetup import generateSetup
from functionsPilotAlloc import pilotAssignment
from functionsPilotAllocRandom import pilotAssignmentRandom
from functionsComputeNMSE_uplink import functionComputeNMSE_uplink
import math

def save_dataset_to_csv(filename, APpositions, UEpositions, best_pilot_index, system_NMSE, UEs_NMSE, mode):
    """
    Guarda las variables relevantes en un archivo CSV para su uso posterior en el entrenamiento de la red neuronal.

    :param filename: Nombre del archivo donde se guardarán los datos.
    :param APpositions: Posiciones de los puntos de acceso (AP).
    :param UEpositions: Posiciones de los usuarios finales (UE).
    :param best_pilot_index: Índices de los pilotos asignados a los UE.
    :param system_NMSE: Suma de los NMSE del sistema.
    :param UEs_NMSE: NMSE individuales de los usuarios.
    :param mode: Tipo de asignación ('optima' o 'random').
    """
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Write header
        writer.writerow(['mode', 'APposition', 'UEposition', 'best_pilot_index', 'system_NMSE', 'UEs_NMSE'])

        for i in range(len(APpositions)):
            # Convertir las posiciones de los APs y los UE a cadenas
            ap_position_str = str(APpositions[i])
            ue_positions_str = ','.join(map(str, UEpositions[i]))
            # Escribir cada fila con los datos separados por comas
            writer.writerow([
                mode,
                ap_position_str,
                ue_positions_str,
                ' '.join(map(str, best_pilot_index[i])),
                str(system_NMSE[i]),
                ' '.join(map(str, UEs_NMSE[i]))
            ])

## Setting Parameters
nbrOfSetups = 5000 # number of Monte-Carlo setups

L = 1                                # number of APs
N = 10                              # number of antennas per AP

# configurables
K =  7                # number of UEs
tau_p = 5           # length of pilot sequences
# configurables


ASD_varphi = math.radians(10)        # Azimuth angle - Angular Standard Deviation in the local scattering model
p = 100                              # total uplink transmit power per UE

# Initialize arrays to store all data
allAPposition_opt = np.zeros((nbrOfSetups, 1), dtype=complex)
allUEpositions_opt = np.zeros((nbrOfSetups, K), dtype=complex)
allBest_pilot_index_opt = np.zeros((nbrOfSetups, K), dtype=int)
allSystem_NMSE_opt = np.zeros((nbrOfSetups,))
allUEs_NMSE_opt = np.zeros((nbrOfSetups, K))


allAPposition_rand = np.zeros((nbrOfSetups, 1), dtype=complex)
allUEpositions_rand = np.zeros((nbrOfSetups, K), dtype=complex)
allBest_pilot_index_rand = np.zeros((nbrOfSetups, K), dtype=int)
allSystem_NMSE_rand = np.zeros((nbrOfSetups,))
allUEs_NMSE_rand = np.zeros((nbrOfSetups, K))

# iterate over the setups
for iter in range(nbrOfSetups):
    print("Setup iteration {} of {}".format(iter, nbrOfSetups))

    # Generate one setup with UEs and APs at random locations
    gainOverNoisedB, R, APpositions, UEpositions = generateSetup(L, K, N, tau_p, ASD_varphi, seed=iter)


    # Asignación óptima de pilotos
    optimal_pilotIndex = pilotAssignment(K, L, N, tau_p, APpositions, UEpositions, gainOverNoisedB, R)
    random_pilotIndex = pilotAssignmentRandom(K, L, N, tau_p, APpositions, UEpositions, gainOverNoisedB, R, mode='random')

    # Compute NMSE for all the UEs
    system_NMSE_opt, UEs_NMSE_opt = functionComputeNMSE_uplink(tau_p, N, K, L, R, optimal_pilotIndex)
    system_NMSE_rand, UEs_NMSE_rand = functionComputeNMSE_uplink(tau_p, N, K, L, R, random_pilotIndex)

    # Append data to arrays for optimal
    allAPposition_opt[iter] = APpositions
    allUEpositions_opt[iter] = np.squeeze(UEpositions)
    allBest_pilot_index_opt[iter] = optimal_pilotIndex
    allSystem_NMSE_opt[iter] = system_NMSE_opt
    allUEs_NMSE_opt[iter] = UEs_NMSE_opt

    

    # Append data to arrays for random
    allAPposition_rand[iter] = APpositions
    allUEpositions_rand[iter] = np.squeeze(UEpositions)
    allBest_pilot_index_rand[iter] = random_pilotIndex
    allSystem_NMSE_rand[iter] = system_NMSE_rand
    allUEs_NMSE_rand[iter] = UEs_NMSE_rand

# Save dataset with all results for optimal
save_dataset_to_csv('dataset_optimal_7_3_p_50m2.csv', allAPposition_opt, allUEpositions_opt, allBest_pilot_index_opt, allSystem_NMSE_opt, allUEs_NMSE_opt, mode='optimal')
print('Optimal assignments saved.')

# Save dataset with all results for random
save_dataset_to_csv('dataset_random_7_3_p_50m2.csv', allAPposition_rand, allUEpositions_rand, allBest_pilot_index_rand, allSystem_NMSE_rand, allUEs_NMSE_rand,mode='random')
print('Random assignments saved.')

rand = np.mean(system_NMSE_rand)
opt = np.mean(system_NMSE_opt)
print(f'Average NMSE for rand assignments (from data): {rand}')
print(f'Average NMSE for optimal assignments (from data): {opt}')


