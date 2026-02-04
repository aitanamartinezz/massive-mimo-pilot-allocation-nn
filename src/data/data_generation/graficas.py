import matplotlib.pyplot as plt
import numpy as np


## GRÁFICO 1: COMPARATIVA DE NMSE PARA DIFERENTES MÉTODOS DE ASIFNACION EN FUNCIÓN DEL Nº DE USUARIOS


# Datos
usuarios = [6, 7, 8, 9, 10, 11, 12]
optima = [0.2606, 0.5783, 0.9360, 1.3547, 1.7774, 2.3763, 3.0143]
random = [1.2254, 1.7570, 2.2784, 3.1520, 3.6936, 4.2841, 5.0134]
kbeams_optima = [0.4568, 0.7925, 1.2347, 1.7496, 2.2523, 2.8857, 3.5525]
prediccion = [0.4298, 0.7789, 1.1211, 1.5561, 1.9888, 2.6499, 3.2312]

# Crear la figura y el gráfico
plt.figure(figsize=(10, 6))

# Graficar cada método con colores y estilos potentes
plt.plot(usuarios, optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="green", label="Óptima")
plt.plot(usuarios, random, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="red", label="Aleatoria")
plt.plot(usuarios, kbeams_optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="#FF7F00", label="KBeams-Óptima")
plt.plot(usuarios, prediccion, marker='^', linestyle='--', linewidth=2.5, markersize=10, color="darkblue", label="Predicción")

# Personalizar el gráfico
plt.xlabel("Número de Usuarios", fontsize=14)
plt.ylabel("NMSE", fontsize=14)
plt.title("Comparación de NMSE para diferente número de usuarios", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



# Configuración personalizada de ticks en el eje Y para que muestre valores enteros
plt.yticks([0.2, 0.5, 1, 2, 3, 4, 5], fontsize=12)

# Ajustes de ticks en el eje X para mejorar legibilidad
plt.xticks(usuarios, fontsize=12)

# Añadir anotaciones solo en la serie de predicción
for x, y in zip(usuarios, prediccion):
    plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, color="darkblue")

plt.grid(True)
# Mostrar el gráfico
plt.show()


## GRÁFICA 2: POR Nº DE PORTADORAS

# Datos
portadoras = [2, 3, 4, 5, 6]
optima = [3.7001, 1.7774, 0.8381, 0.2871, 0.1227]
random = [4.6276, 3.6936, 2.5327,2.0321, 1.6745]
kbeams_optima = [4.0780, 2.2523, 1.2018, 0.6338, 0.2755]
prediccion = [3.9012, 1.9888, 1.0100, 0.4612, 0.2612]

# Crear la figura y el gráfico
plt.figure(figsize=(10, 6))

# Graficar cada método con colores y estilos potentes
plt.plot(portadoras, optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="green", label="Óptimo")
plt.plot(portadoras, random, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="red", label="Aleatorio")
plt.plot(portadoras, kbeams_optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="#FF7F00", label="KBeams-Óptimo")
plt.plot(portadoras, prediccion, marker='^', linestyle='--', linewidth=2.5, markersize=10, color="darkblue", label="Predicción")

# Personalizar el gráfico
plt.xlabel("Número de portadoras", fontsize=14)
plt.ylabel("NMSE", fontsize=14)
plt.title("Comparación de NMSE para diferente número de portadoras", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



# Configuración personalizada de ticks en el eje Y para que muestre valores enteros
plt.yticks([0.1, 0.5, 1, 2, 3, 4, 5], fontsize=12)

# Ajustes de ticks en el eje X para mejorar legibilidad
plt.xticks(portadoras, fontsize=12)

# Añadir anotaciones solo en la serie de predicción
for x, y in zip(portadoras, prediccion):
    plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, color="darkblue")

plt.grid(True)
# Mostrar el gráfico
plt.show()



## GRÁFICA 2: POR metros cuadrados

# Datos
portadoras = [10, 20, 50, 100, 150]
optima = [1.6864, 1.7293, 1.8445, 1.9837, 2.1604]
random = [3.4079, 3.4569, 3.6936,3.8634, 4.1101]
kbeams_optima = [1.9967, 2.0528, 2.2523, 2.4082, 2.4722]
prediccion = [1.8823, 1.9833, 2.0867, 2.2312, 2.3456]

# Crear la figura y el gráfico
plt.figure(figsize=(10, 6))

# Graficar cada método con colores y estilos potentes
plt.plot(portadoras, optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="green", label="Óptima")
plt.plot(portadoras, random, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="red", label="Aleatoria")
plt.plot(portadoras, kbeams_optima, marker='o', linestyle='-', linewidth=2.5, markersize=10, color="#FF7F00", label="KBeams-Óptima")
plt.plot(portadoras, prediccion, marker='^', linestyle='--', linewidth=2.5, markersize=10, color="darkblue", label="Predicción")

# Personalizar el gráfico
plt.xlabel("Metros", fontsize=14)
plt.ylabel("NMSE", fontsize=14)
plt.title("Comparación de NMSE en función de la extensión del área desde el AP", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)



# Configuración personalizada de ticks en el eje Y para que muestre valores enteros
plt.yticks([0.1, 0.5, 1, 2, 3, 4, 5], fontsize=12)

# Ajustes de ticks en el eje X para mejorar legibilidad
plt.xticks(portadoras, fontsize=12)

# Añadir anotaciones solo en la serie de predicción
for x, y in zip(portadoras, prediccion):
    plt.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, color="darkblue")

plt.grid(True)
# Mostrar el gráfico
plt.show()


