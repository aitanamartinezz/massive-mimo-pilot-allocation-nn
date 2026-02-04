import pandas as pd
import random

# Cargar los tres datasets de los escenarios
df1 = pd.read_csv('dataset_optimal_5_2_p_20m2.csv')
df2 = pd.read_csv('dataset_optimal_7_3_p_20m2.csv')
df3 = pd.read_csv('dataset_optimal_9_2_p_20m2.csv')

# Agregar una columna para identificar el escenario de origen
df1['Escenario'] = 'Escenario 1'
df2['Escenario'] = 'Escenario 2'
df3['Escenario'] = 'Escenario 3'

# Concatenar los tres datasets
df_combined = pd.concat([df1, df2, df3])

# Crear una columna identificadora única que combine los valores de todas las columnas
df_combined['ID'] = df_combined.astype(str).agg('-'.join, axis=1)

# Agrupar por la nueva columna 'ID' que ahora identifica de manera única cada registro
df_grouped = df_combined.groupby('ID')

# Seleccionar aleatoriamente un escenario para cada grupo de registros
df_randomized = df_grouped.apply(lambda x: x.sample(n=1))

# Restablecer el índice para limpiar la salida
df_randomized.reset_index(drop=True, inplace=True)

# Guardar el nuevo dataset combinado con escenarios aleatorios
df_randomized.to_csv('dataset_aleatorio.csv', index=False)

print("Combinación aleatoria completa y guardada.")