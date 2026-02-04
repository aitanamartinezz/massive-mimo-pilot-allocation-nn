import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

# Cargar el archivo original
file_path = 'dataset_optimal_5_2_p_50m2.csv'
original_data = pd.read_csv(file_path)

# Parámetros de configuración
num_new_scenarios = 10000  # Número total de nuevos escenarios a generar
position_variation = 0.05  # Variación aleatoria máxima en posiciones de UE
nmse_variation_factor = 0.1  # Factor de variación en NMSE
num_strata = 10  # Número de estratos que quieres definir
proximity_threshold = 6.75  # Umbral de proximidad para considerar posiciones similares
nmse_tolerance = 0.5  # Tolerancia para el NMSE
max_attempts = 15  # Máximo de intentos antes de regenerar las posiciones


def definir_estratos(data, num_strata):
    """Define estratos basados en la posición de los UEs."""
    estratos = []
    for index, row in data.iterrows():
        ue_positions = eval(row['UEposition'])
        coords = np.array([pos.real for pos in ue_positions])
        if len(coords) > 0:
            estrato = pd.cut(coords, bins=num_strata, labels=False)
            estratos.append(estrato[0])
    if len(estratos) == len(data):
        data['estrato'] = estratos
    else:
        print("Error: La longitud de los estratos no coincide con el número de filas del DataFrame.")
    return data


# Definir los estratos
original_data = definir_estratos(original_data, num_strata)


# Función para muestreo estratificado
def muestreo_estratificado(data, num_samples):
    estratos = data['estrato'].unique()
    estratos_muestreados = []
    for estrato in estratos:
        estrato_data = data[data['estrato'] == estrato]
        num_samples_per_stratum = num_samples // len(estratos)
        if len(estrato_data) >= num_samples_per_stratum:
            sampled_data = estrato_data.sample(n=num_samples_per_stratum, random_state=42)
            estratos_muestreados.append(sampled_data)
    return pd.concat(estratos_muestreados)


# Realizar muestreo estratificado
sampled_data = muestreo_estratificado(original_data, num_new_scenarios)

# Almacenar asignaciones previas
previous_assignments = {}


# Función para calcular la distancia euclidiana
def calcular_distancia(pos1, pos2):
    return np.abs(pos1 - pos2)


# Función para verificar la distancia mínima
def verificar_distancia_minima(positions, assignment, min_distance=6.75):
    distances = pdist(np.array([[pos.real, pos.imag] for pos in positions]))
    return np.all(distances >= min_distance)


# Función para buscar asignación similar
def buscar_asignacion_similar(ue_positions, nmse_actual):
    """Busca una asignación previa similar basándose en las posiciones y el NMSE."""
    for key, (pos, assignment, nmse) in previous_assignments.items():
        if np.all([calcular_distancia(p, q) < proximity_threshold for p, q in zip(ue_positions, pos)]):
            if np.abs(nmse_actual - nmse) < nmse_tolerance:
                #print(f"Asignación similar encontrada para posiciones {ue_positions} -> {assignment}")
                return assignment
    return None


# Listas para almacenar nuevos datos
new_rows = []

print("Iniciando generación de escenarios...")

for i in range(num_new_scenarios):
    attempts = 0
    while attempts < max_attempts:
        base_row = sampled_data.sample(n=1).iloc[0]
        new_row = base_row.copy()
        ue_positions = eval(base_row['UEposition'])

        # Variar ligeramente las posiciones de los UEs
        new_ue_positions = [
            complex(pos.real + np.random.uniform(-position_variation, position_variation),
                    pos.imag + np.random.uniform(-position_variation, position_variation))
            for pos in ue_positions
        ]

        new_row['UEposition'] = ','.join(str(pos) for pos in new_ue_positions)
        new_row['system_NMSE'] = base_row['system_NMSE'] * (
                    1 + np.random.uniform(-nmse_variation_factor, nmse_variation_factor))

        # Ajustar el NMSE de cada UE
        ues_nmse = list(map(float, str(base_row['UEs_NMSE']).split()))
        new_ues_nmse = [
            nmse * (1 + np.random.uniform(-nmse_variation_factor, nmse_variation_factor))
            for nmse in ues_nmse
        ]
        new_row['UEs_NMSE'] = ' '.join(map(str, new_ues_nmse))

        # Intentar encontrar una asignación similar
        asignacion_similar = buscar_asignacion_similar(new_ue_positions, new_row['system_NMSE'])

        if asignacion_similar is not None:
            new_row['best_pilot_index'] = ' '.join(map(str, asignacion_similar))
            print(f"Escenario {i + 1}: Asignación similar encontrada tras {attempts + 1} intentos.")
            break
        else:
            attempts += 1
            print(f"Intento {attempts} sin asignación similar, generando nuevas posiciones.")

            # Regenerar nuevas posiciones si el número de intentos máximo es alcanzado
            if attempts == max_attempts:
                ue_positions = [complex(np.random.uniform(0, 20), np.random.uniform(0, 20)) for _ in
                                range(len(ue_positions))]
                print(
                    f"Escenario {i + 1}: Asignación no encontrada tras {max_attempts} intentos. Regenerando posiciones completas.")

    # Si se encontró una asignación o al crear nuevas posiciones
    previous_assignments[tuple(new_ue_positions)] = (
    new_ue_positions, list(map(int, new_row['best_pilot_index'].split())), new_row['system_NMSE'])
    new_rows.append(new_row)

print("Generación de escenarios completada.")

# Crear un nuevo DataFrame con los escenarios originales y generados
augmented_data = pd.concat([original_data, pd.DataFrame(new_rows)], ignore_index=True)

# Guardar el DataFrame en un nuevo archivo CSV
augmented_file_path = '2.csv'
augmented_data.to_csv(augmented_file_path, index=False)

print(f"Data augmentation completado. Archivo guardado en: {augmented_file_path}")
