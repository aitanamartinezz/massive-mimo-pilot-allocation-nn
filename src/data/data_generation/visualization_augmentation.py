import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Cargar el archivo CSV con los datos generados
augmented_data = pd.read_csv('pruebas.csv')

# Configuración para la visualización
colors = {0: 'red', 1: 'green', 2: 'blue'}  # Colores para las portadoras

# Determinar el número de escenarios originales en el dataset
num_original_scenarios = len(pd.read_csv('dataset_optimal_7_3_20m2.csv'))

# Filtrar solo los nuevos escenarios
new_scenarios = augmented_data[num_original_scenarios:]

# Número de nuevos escenarios a visualizar
num_new_scenarios_to_visualize = 40  # Cambiar según sea necesario

# Posición de la Antena (AP)
ap_position = (25, 25)

# Visualizar los nuevos escenarios
for index in range(num_new_scenarios_to_visualize):
    # Obtener una fila de un nuevo escenario
    row = new_scenarios.iloc[index]

    # Obtener posiciones de UEs
    ue_positions = eval(row['UEposition'])

    # Obtener asignaciones de portadoras
    best_pilot_index = list(map(int, row['best_pilot_index'].split()))

    # Crear gráfico
    plt.figure(figsize=(8, 8))

    # Dibujar posiciones de UEs
    for i, pos in enumerate(ue_positions):
        plt.scatter(pos.real, pos.imag, color=colors[best_pilot_index[i]], s=100,
                    label=f'UE {i + 1} (Pilot {best_pilot_index[i]})')

    # Dibujar la posición de la antena
    plt.scatter(ap_position[0], ap_position[1], color='black', s=200, label='Antena (AP)', marker='X')

    # Configuración del gráfico
    plt.title(f'Nuevo Escenario {index + 1}')
    plt.xlabel('Posición X (m)')
    plt.ylabel('Posición Y (m)')
    plt.axhline(0, color='black', linewidth=0.5, ls='--')
    plt.axvline(0, color='black', linewidth=0.5, ls='--')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()

    # Ajustar límites a 0-50
    plt.xlim(0, 50)
    plt.ylim(0, 50)

    plt.gca().set_aspect('equal', adjustable='box')

    # Mostrar gráfico
    plt.show()



