# predict.py

import numpy as np

# [Definiciones de funciones y clases aquí, igual que tu código inicial]
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Clase de capa oculta (si es que no la importas desde otro lugar)
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons, n_inputs)
        self.biases = np.zeros((n_neurons, 1))

# Inicializar capas (con dimensiones adecuadas)
hidden_layer = Layer_Dense(n_inputs=784, n_neurons=32)
output_layer = Layer_Dense(n_inputs=32, n_neurons=10)

# Cargar pesos y sesgos
hidden_layer.weights = np.load("data/hidden_weights.npy")
output_layer.weights = np.load("data/output_weights.npy")
hidden_layer.biases = np.load("data/hidden_biases.npy")

# Función para realizar predicciones
def make_prediction(input_):
    z1 = hidden_layer.weights @ input_ + hidden_layer.biases
    a1 = relu(z1)
    z2 = output_layer.weights @ a1
    a2 = softmax(z2).T
    return np.argmax(a2)
