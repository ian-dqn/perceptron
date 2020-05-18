import numpy as np

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

train_inputs = np.array([[0,0,1],
			 [1,1,1],
			 [1,0,1],
			 [0,1,1]])

train_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights:')
print(synaptic_weights)

for i in range(20000):
  input_layer = train_inputs
  outputs = sigmoid(np.dot(input_layer, synaptic_weights))
  error = train_outputs - outputs
  adjustements = error *sigmoid_derivative(outputs)
  synaptic_weights += np.dot(input_layer.T, adjustements)

print('Synaptic weights after training')
print(synaptic_weights)

print('outputs after training:')
print(outputs)
