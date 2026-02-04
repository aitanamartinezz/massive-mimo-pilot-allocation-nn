import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kerastuner as kt

# Cargar los datos desde el archivo CSV
datos = pd.read_csv("data_generation/dataset500000_5_2_20m2.csv")

# Extraer las posiciones de los usuarios
escenarios = datos["UEposition"]

# Descomposición de números complejos en partes reales e imaginarias
def descomponer_escenarios(escenarios):
    descompuestos = []
    for escenario in escenarios:
        usuarios = escenario.split('),(')
        usuarios[0] = usuarios[0].replace('(', '')
        usuarios[-1] = usuarios[-1].replace(')', '')
        descompuesto = []
        for usuario in usuarios:
            usuario = complex(usuario)
            descompuesto.append(usuario.real)
            descompuesto.append(usuario.imag)
        descompuestos.append(descompuesto)
    return np.array(descompuestos)

x_data = descomponer_escenarios(escenarios)

# Extraer y procesar las asignaciones de portadoras
def procesar_portadoras(portadoras_str):
    return list(map(int, portadoras_str.split()))

portadoras = datos['best_pilot_index'].apply(procesar_portadoras).tolist()
portadoras = np.array(portadoras)

# División de los datos en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_data, portadoras, test_size=0.2, random_state=42)

# Normalización de los datos
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Convertir y_train y y_test a categórico para clasificación
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# Reshape data for Dense layers
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

# Definir la función de construcción del modelo para Keras Tuner
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(x_train.shape[1],)))

    # Añadir exactamente 3 capas densas
    for i in range(8):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=64, max_value=1024, step=64),
                        activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.1, max_value=0.5, step=0.1)))

    # Capa de salida
    model.add(Dense(10, activation='softmax'))
    model.add(Reshape((5, 2)))

    # Compilar el modelo
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Inicializar el tuner
tuner = kt.Hyperband(build_model,
                     objective=kt.Objective('val_accuracy', direction='max'),
                     max_epochs=10,
                     factor=3,
                     directory='tuner',
                     project_name='tuner_validation_8')


# Definir callbacks
stop_early = EarlyStopping(monitor='val_loss', patience=5)

# Realizar la búsqueda de hiperparámetros para val_accuracy
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[stop_early])


# Obtener los mejores hiperparámetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


print(f"""
El número óptimo de unidades en cada capa es (val_accuracy):
""")
for i in range(4):
    print(
        f"Capa {i + 1}: {best_hps.get('units_' + str(i))} unidades con un dropout de {best_hps.get('dropout_' + str(i))}")



# Construir el mejor modelo con los mejores hiperparámetros para val_accuracy
model = tuner.hypermodel.build(best_hps)
history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

# Evaluar el modelo para val_accuracy
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy (val_accuracy): {accuracy * 100:.2f}%')




# Guardar el mejor modelo
model.save('best_model_tuned_accuracy.keras')
model_loss.save('best_model_tuned_loss.keras')


