import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Reshape
from tensorflow.keras.models import Sequential
from sklearn.metrics import roc_curve, auc
from itertools import cycle


# ==============================
# FUNCIONES DE CARGA Y PROCESAMIENTO DE DATOS
# ==============================
def cargar_y_procesar_datos(ruta_csv):
    datos = pd.read_csv(ruta_csv)
    escenarios = datos["UEposition"]
    nmse = datos["system_NMSE"]

    # Descomponer la posición de los usuarios
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

    # Procesar las portadoras
    def procesar_portadoras(portadoras_str):
        return list(map(int, portadoras_str.split()))

    portadoras = datos['best_pilot_index'].apply(procesar_portadoras).tolist()
    portadoras = np.array(portadoras)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    x_train, x_test, y_train, y_test, nmse_train, nmse_test = train_test_split(x_data, portadoras, nmse, test_size=0.2
                                                                                       , random_state=42)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]))

    return x_train, x_test, y_train, y_test, nmse_train, nmse_test

x_train_opt, x_test_opt, y_train_opt, y_test_opt, nmse_train_opt, nmse_test_opt = cargar_y_procesar_datos("dataset_optimal_5_2_p_20m2.csv")

# ==============================
# DEFINICIÓN DE LOS MODELOS
# ==============================
def create_model_2_9(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=(input_shape)))

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

    model.add(Dense(18, activation='softmax'))
    model.add(Reshape((9, 2)))

    return model





def create_model_5_2(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(900, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1400, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(900, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(500, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))
    model.add(Reshape((5, 2)))

    return model



def create_model_7_3(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Primera capa densa con menos neuronas
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Segunda capa densa
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Tercera capa densa
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Cuarta capa densa con menos neuronas
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Quinta capa densa
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Capa de salida
    model.add(Dense(21, activation='softmax'))
    model.add(Reshape((7, 3)))

    return model


def create_model_7_3_malo(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))



    # Segunda capa densa
    model.add(Dense(248, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Tercera capa densa
    model.add(Dense(504, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(600, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))




    # Quinta capa densa
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    # Quinta capa densa
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(21, activation='softmax'))
    model.add(Reshape((7, 3)))

    return model

def create_model_6_4(input_shape):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    # Primera capa densa con menos neuronas
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Segunda capa densa
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Tercera capa densa
    model.add(Dense(2048, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # Cuarta capa densa con menos neuronas
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Quinta capa densa
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # Capa de salida
    model.add(Dense(24, activation='softmax'))
    model.add(Reshape((6, 4)))

    return model

# ==============================
# CREAR Y COMPILAR EL MODELO
# ==============================
model_opt = create_model_6_4((x_train_opt.shape[1],))
optimizer_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model_opt.compile(optimizer=optimizer_opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks para entrenamiento
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
checkpoint_opt = ModelCheckpoint('best_model_optimal.keras', monitor='val_accuracy', save_best_only=True, mode='max')

# ==============================
# ENTRENAMIENTO DEL MODELO
# ==============================
print("Entrenamiento óptimo:")
history_opt = model_opt.fit(x_train_opt, y_train_opt, epochs=20, batch_size=128,
                            validation_data=(x_test_opt, y_test_opt),
                            callbacks=[early_stopping, reduce_lr, checkpoint_opt])


# Evaluación del modelo
loss_opt, accuracy_opt = model_opt.evaluate(x_test_opt, y_test_opt)
print(f'Optimal Dataset - Accuracy: {accuracy_opt * 100:.2f}%')

# ==============================
# PREDICCIONES Y MATRICES DE CONFUSIÓN
# ==============================
# Predicciones del modelo
predicciones_opt = model_opt.predict(x_test_opt)
predicciones_asignaciones_opt = np.argmax(predicciones_opt, axis=2)

# Asignaciones óptimas reales
asignaciones_optimas = np.argmax(y_test_opt, axis=2)

# Visualización de resultados
print("Asignaciones óptimas:")
print("Ejemplos de predicciones vs. etiquetas reales:")
for i in range(10):
    print(f"Predicción {i + 1}: {predicciones_asignaciones_opt[i]}")
    print(f"Óptima      {i + 1}: {np.argmax(y_test_opt[i], axis=1)}")
    #print(f"RANDOM     {i + 1}: {np.argmax(y_test_rand[i], axis=1)}")

# ==============================
# MATRIZ DE CONFUSIÓN ORIGINAL
# ==============================
cm = confusion_matrix(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten())
print("Matriz de confusión (original):")
print(cm)

# Visualizar la matriz de confusión original
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=True)
plt.title("Matriz de Confusión Original")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# ==============================
# MATRIZ DE CONFUSIÓN MEJORADA
# ==============================
# Reporte detallado de las métricas de evaluación
print("Reporte de clasificación:")
print(classification_report(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten()))


# Crear la visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Predicho 0', 'Predicho 1'],
            yticklabels=['Real 0', 'Real 1'])

plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()
# ==============================
# CÁLCULO DE MÉTRICAS ADICIONALES
# ==============================
# Cálculo de precisión, recall y F1 para micro, macro y weighted
precision_micro = precision_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='micro')
recall_micro = recall_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='micro')
f1_micro = f1_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='micro')

precision_macro = precision_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='macro')
recall_macro = recall_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='macro')
f1_macro = f1_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='macro')

precision_weighted = precision_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='weighted')
recall_weighted = recall_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='weighted')
f1_weighted = f1_score(asignaciones_optimas.flatten(), predicciones_asignaciones_opt.flatten(), average='weighted')

# Mostrar las métricas
print(f"Micro Precision: {precision_micro:.4f}, Micro Recall: {recall_micro:.4f}, Micro F1: {f1_micro:.4f}")
print(f"Macro Precision: {precision_macro:.4f}, Macro Recall: {recall_macro:.4f}, Macro F1: {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}, Weighted Recall: {recall_weighted:.4f}, Weighted F1: {f1_weighted:.4f}")

# ==============================
# GRÁFICOS DE ACCURACY Y LOSS
# ==============================
import matplotlib.pyplot as plt
import numpy as np

# Si tienes seaborn instalado, usa esta línea:
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except ImportError:
    print("Seaborn no está disponible, usando estilo predeterminado.")
    plt.style.use('ggplot')  # Alternativa

# Simulación de varias ejecuciones de entrenamiento (opcional)
train_accuracies = [history_opt.history['accuracy']]
val_accuracies = [history_opt.history['val_accuracy']]
train_losses = [history_opt.history['loss']]
val_losses = [history_opt.history['val_loss']]

# Calcular medias y desviaciones estándar
mean_train_accuracy = np.mean(train_accuracies, axis=0)
std_train_accuracy = np.std(train_accuracies, axis=0)
mean_val_accuracy = np.mean(val_accuracies, axis=0)
std_val_accuracy = np.std(val_accuracies, axis=0)
mean_train_loss = np.mean(train_losses, axis=0)
std_train_loss = np.std(train_losses, axis=0)
mean_val_loss = np.mean(val_losses, axis=0)
std_val_loss = np.std(val_losses, axis=0)

# Definir épocas
epochs = range(1, len(mean_train_accuracy) + 1)

# Subplots de Accuracy y Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico de Accuracy
ax1.plot(epochs, mean_train_accuracy, 'bo-', label='Training Accuracy')
ax1.fill_between(epochs, mean_train_accuracy - std_train_accuracy, mean_train_accuracy + std_train_accuracy, color='b', alpha=0.2)
ax1.plot(epochs, mean_val_accuracy, 'ro-', label='Validation Accuracy')
ax1.fill_between(epochs, mean_val_accuracy - std_val_accuracy, mean_val_accuracy + std_val_accuracy, color='r', alpha=0.2)
ax1.set_title("Training and Validation Accuracy")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True)

# Gráfico de Loss
ax2.plot(epochs, mean_train_loss, 'bo-', label='Training Loss')
ax2.fill_between(epochs, mean_train_loss - std_train_loss, mean_train_loss + std_train_loss, color='b', alpha=0.2)
ax2.plot(epochs, mean_val_loss, 'ro-', label='Validation Loss')
ax2.fill_between(epochs, mean_val_loss - std_val_loss, mean_val_loss + std_val_loss, color='r', alpha=0.2)
ax2.set_title("Training and Validation Loss")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True)

# Mostrar y guardar la figura
plt.tight_layout()
plt.savefig("training_validation_metrics.png", dpi=300)
plt.show()


# ==============================
# GRÁFICO DE F1-SCORE
# ==============================
# Creación del gráfico de F1 Score para micro, macro y weighted

labels = ['Micro F1', 'Macro F1', 'Weighted F1']
f1_scores = [f1_micro, f1_macro, f1_weighted]

plt.figure(figsize=(8, 5))
plt.bar(labels, f1_scores, color=['#5b9bd5', '#ed7d31', '#a5a5a5'])
plt.title("F1 Score: Micro, Macro y Weighted")
plt.xlabel("Tipo de F1 Score")
plt.ylabel("Valor de F1 Score")
plt.ylim(0, 1)  # Ajustar el rango del eje Y de 0 a 1 para mejor visualización
plt.show()

