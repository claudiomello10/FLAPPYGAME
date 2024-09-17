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
        self.fc1 = ComplexLinear(7, 4)  # Input layer (784 inputs, 128 hidden units)
        self.fc2 = ComplexLinear(4, 2)  # Hidden layer (128 inputs, 64 hidden units)
        self.fc3 = ComplexLinear(2, 1)  # Output layer (64 inputs, 10 outputs)

    def forward(self, x):
        x = self.fc1(x)  # First hidden layer
        x = complex_relu(x)  # Apply ReLU activation to the first hidden layer
        x = self.fc2(x)  # Second hidden layer
        x = complex_relu(x)  # Apply ReLU activation to the second hidden layer
        x = self.fc3(x)  # Output layer (no activation function)
        return x


# Define the loss function
class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predictions, targets):
        real = torch.mean((predictions.real - targets.real) ** 2)
        imag = torch.mean((predictions.imag - targets.imag) ** 2)
        return torch.sqrt(real + imag).type(torch.float64)


loss_function = nn.L1Loss().type(torch.complex128)
loss_function1 = CustomMSELoss().type(torch.complex128)

# Create an instance of the neural network
model = NeuralNetwork()

# Create an instance of the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10000
input_data = torch.complex(torch.randn(7), torch.randn(7))
target_data = torch.complex(torch.randn(1), torch.tensor(0.4247329842347)).type(torch.complex128)

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
