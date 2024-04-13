import torch
import pygad.torchga as ga
import torch.nn as nn
from complexPyTorch.complexLayers import (
    ComplexBatchNorm2d,
    ComplexConv2d,
    ComplexLinear,
)
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d


# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
        self.fc2 = nn.Conv1d(in_channels=2, out_channels=2, kernel_size=2, stride=2)
        self.fc3 = nn.Linear(1, 2)

    def forward(self, x):
        x = self.fc1(x)  # First hidden layer
        nn.functional.relu(x)  # Apply ReLU activation to the first hidden layer
        x = self.fc2(x)  # Second hidden layer
        nn.functional.relu(x)
        x = self.fc3(x)  # Output layer (no activation function)
        return x


loss_function = nn.MSELoss()

# Create an instance of the neural network
model = NeuralNetwork()

# Create an instance of the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10000
input_data = torch.randn(1, 2, 7)
target_data = torch.randn(1, 2)

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    output = model(input_data)

    # Compute the loss
    loss = loss_function(output, target_data)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss
    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


# Get the model output
output = model(input_data)


print(f"output: {output} \n \n")  # Print the output of the neural network

print(f"target_data: {target_data} \n \n")  # Print the target data
