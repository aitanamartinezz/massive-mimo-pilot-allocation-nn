
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Reshape
from tensorflow.keras.models import Sequential

def create_model_general(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Capas densas comunes para todas las configuraciones
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # No definimos el número de neuronas de salida aquí, se hará dinámicamente al cargar el dataset

    return model


# Modificamos la función de entrenamiento para ajustar la capa de salida a los datos
def entrenar_modelo(x_train, y_train, x_test, y_test, num_users, num_portadoras):
    model = create_model_general((x_train.shape[1],))

    # Añadimos la capa de salida dinámica según los usuarios y portadoras
    model.add(Dense(num_users * num_portadoras, activation='softmax'))
    model.add(Reshape((num_users, num_portadoras)))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max')

    # Entrenamiento
    print(f"Entrenando el modelo con {num_users} usuarios y {num_portadoras} portadoras")
    history = model.fit(x_train, y_train, epochs=10, batch_size=50, validation_data=(x_test, y_test),
                        callbacks=[early_stopping, reduce_lr, checkpoint])

    # Evaluación del modelo
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Dataset - Usuarios: {num_users}, Portadoras: {num_portadoras}, Accuracy: {accuracy * 100:.2f}%')

    return model, history

