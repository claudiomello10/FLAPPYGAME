import torch

import torch.nn as nn


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(7, 4)  # Input layer (784 inputs, 128 hidden units)
        self.fc2 = nn.Linear(4, 2)  # Hidden layer (128 inputs, 64 hidden units)
        self.fc3 = nn.Linear(2, 1)  # Output layer (64 inputs, 10 outputs)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first hidden layer
        x = torch.relu(self.fc2(x))  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)  # Output layer (no activation function)
        return x


# Create an instance of the neural network
model = NeuralNetwork()

# Print the model architecture
print(model)


# Get the model weights
weights = model.state_dict()


# Print the model weights
print(weights["fc1.weight"].shape)
