import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
LEARNING_RATE = 0.01
EPOCHS = 1
np.random.seed(7) #agregamos una semilla

df = pd.read_csv('data/mnist_train.csv')

imgs =df.iloc[0:,1:] #separamos las imagenes de los labels
X=imgs.to_numpy().T
y = df.iloc[:,0:1] #capturamos los labels
y = y.to_numpy().squeeze()
unique, counts = np.unique(y, return_counts=True)

y_train = np.zeros((10,len(y)))
for i, y_it in enumerate(y):
    y_train[y_it][i] = 1
#Normalizing data to prevent overflow
X=X/255

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

#Creating model
#2 hidden layers
hidden_layer = Layer_Dense(n_inputs=784,n_neurons=32) 
output_layer=Layer_Dense(n_inputs=32,n_neurons=10)
print("training in process...")
#Training #MODIFY!
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
print("Neural network trained sucessfully!")
data_test=pd.read_csv("data/mnist_test.csv")
x_test=data_test.iloc[:,1:].to_numpy().T/255
y_test=data_test.iloc[:,:1].to_numpy().T.squeeze()
print(x_test.shape)
print(y_test.shape)
def prediction(input_,y_true,n_cases):
    hits=0
    for i in range(n_cases):
        z1=hidden_layer.weights @ input_[:,i:i+1] + hidden_layer.biases
        a1=relu(z1)
        z2=output_layer.weights @ a1
        a2=softmax(z2).T
        pred=np.argmax(a2)
        print(f"Prediction: {pred} vs Real: {y_true[i]}",end=" ")
        if pred==y_true[i]:
            hits+=1
            print("CORRECT")
        else:
            print("INCORRECT")
    return "Precission: "+str(hits/n_cases)
print(prediction(x_test,y_test,100))