# train_model.py

import numpy as np
import pandas as pd

LEARNING_RATE = 0.01
EPOCHS = 1
np.random.seed(7)

df = pd.read_csv('data/mnist_train.csv')
imgs = df.iloc[:, 1:]
X = imgs.to_numpy().T
y = df.iloc[:, 0:1].to_numpy().squeeze()

y_train = np.zeros((10, len(y)))
for i, y_it in enumerate(y):
    y_train[y_it][i] = 1
X = X / 255

# [Definiciones de funciones y clases aquí... mismo que tu código inicial]
#Activation functions

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax_derivative(output):
    return output * (1 - output)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cross_entropy_derivative(y_true,y_pred):
    return y_pred-y_true # -log(y_true/y_pred)    

#Layers
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_neurons,n_inputs) #W
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        self.output = np.dot(self.weights,inputs) + self.biases


hidden_layer = Layer_Dense(n_inputs=784, n_neurons=32)
output_layer = Layer_Dense(n_inputs=32, n_neurons=10)

# [Código de entrenamiento aquí... igual que antes]
for j in range(EPOCHS):
    total_loss=0
    for i in range(60000):
        #Forward Propagation
        z_1=hidden_layer.weights @ X[:,i:i+1] + hidden_layer.biases
        a_1=relu(z_1)
        z_2=output_layer.weights @ a_1
        a_2=softmax(z_2)
        #Computation of error
        loss = -np.sum(y_train[:, i:i+1] * np.log(a_2 + 1e-10))  # Adding epsilon to avoid log(0)
        total_loss+=loss
        #Back Propagation
        error_term_output = cross_entropy_derivative(y_train[:,i:i+1],a_2) #dL/dA_2 * dA_2/dZ_2
        error_term_hidden = (output_layer.weights.T @ error_term_output) * relu_derivative(z_1)
        #Updating parameters 
        #WEIGHTS
        output_layer.weights -= LEARNING_RATE * (error_term_output @ a_1.T) #delta @ dZ_2/dW_
        hidden_layer.weights -= LEARNING_RATE * (error_term_hidden @ X[:,i:i+1].T)
        #BIAS
        hidden_layer.biases -= LEARNING_RATE * error_term_hidden
    mean_loss = total_loss/60000
    print(f"epoch: {j+1} , loss: {mean_loss} , process: "+"*"*(int((j/EPOCHS)*20)))

print("Neural network trained successfully!")

# Guardando los pesos y sesgos
np.save("data/hidden_weights.npy", hidden_layer.weights)
np.save("data/output_weights.npy", output_layer.weights)
np.save("data/hidden_biases.npy", hidden_layer.biases)
