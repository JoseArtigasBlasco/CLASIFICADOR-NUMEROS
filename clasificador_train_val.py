from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from keras._tf_keras.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np



(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Visualización de 16 imágenes aleatorias del set de entrenamiento
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
ids_imgs = np.random.randint(0, x_train.shape[0], 16)
for i, ax in enumerate(axes.flat):
    img = x_train[ids_imgs[i], :, :]
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    ax.set_title(y_train[ids_imgs[i]])
plt.suptitle('16 imágenes del set MNIST')
plt.show()

# Pre-procesamiento: aplanar y normalizar las imágenes
X_train = x_train.reshape(x_train.shape[0], -1) / 255.0
X_test = x_test.reshape(x_test.shape[0], -1) / 255.0


# Convertir las etiquetas a representación one-hot
nclasses = 10
Y_train = to_categorical(y_train, nclasses)
Y_test = to_categorical(y_test, nclasses)

# Creación del modelo
np.random.seed(1)
input_dim = X_train.shape[1]
output_dim = Y_train.shape[1]

modelo = Sequential([
    Dense(15, input_dim=input_dim, activation='relu'),
    Dense(output_dim, activation='softmax')
])

# Mostrar el resumen del modelo
modelo.summary()

# Compilación del modelo
sgd = SGD(learning_rate=0.2)
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenamiento del modelo
num_epochs = 50
batch_size = 1024
historia = modelo.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, verbose=2)

# Resultados

# Error y precisión vs iteraciones
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(historia.history['loss'])
ax1.set_title('Pérdida vs. iteraciones')
ax1.set_ylabel('Pérdida')
ax1.set_xlabel('Iteración')

ax2.plot(historia.history['accuracy'])
ax2.set_title('Precisión vs. iteraciones')
ax2.set_ylabel('Precisión')
ax2.set_xlabel('Iteración')

plt.show()

# Calcular la precisión sobre el set de validación
accuracy = modelo.evaluate(X_test, Y_test, verbose=0)
print('Precisión set validación: {:.1f}%'.format(100 * accuracy[1]))

# Realizar predicción sobre el set de validación y mostrar algunos ejemplos de la clasificación resultante
Y_pred = np.argmax(modelo.predict(X_test), axis=1)

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
ids_imgs = np.random.randint(0, X_test.shape[0], 16)
for i, ax in enumerate(axes.flat):
    idx = ids_imgs[i]
    img = X_test[idx, :].reshape(28, 28)
    cat_original = np.argmax(Y_test[idx, :])
    cat_prediccion = Y_pred[idx]

    ax.imshow(img, cmap='gray')
    ax.axis('off')
    color = 'red' if cat_original != cat_prediccion else 'blue'
    ax.set_title(f'"{cat_original}" clasificado como "{cat_prediccion}"', color=color)
plt.suptitle('Clasificación set de validación')
plt.show()



