import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import os
from os import system
import pandas as pd
import csv

sech = lambda x: 1/torch.cosh(x)
torch.sech = sech

def flt_to_str(flt):
    return str(flt).replace(".","_point_")

# Constants for the potential
m = 1.0        # Mass
Omega = 1.0    # Frequency of the harmonic trap
A = 1.0        # Amplitude of the Gaussian potential
sigma = 1.0    # Width of the Gaussian potential
T = 10.0       # Final time
dt = 0.01      # Time step
x_star = 0.5   # Final position sought
v_th = 0.01    # Velocity threshold
t_values = np.linspace(1e-8, T, int(T / dt))
x_start, v_start = 1.0, 0.0 #IC
x_start = float(x_start)

# Define the neural network for xi(t)
class XiModel(nn.Module):
    def __init__(self):
        self.num_units = 4
        super(XiModel, self).__init__()
        # Initialize layers with reduced neurons
        self.fc1 = nn.Linear(1, self.num_units)
        self.fc2 = nn.Linear(self.num_units, self.num_units)
        self.fc3 = nn.Linear(self.num_units, self.num_units)
        self.fc4 = nn.Linear(self.num_units, self.num_units)
        self.fc_out = nn.Linear(self.num_units, 1)
        self.fc_skip = nn.Linear(1, 1)
        
    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output_main = self.fc_out(x)
        output_skip = self.fc_skip(inputs)
        return output_main + output_skip

# Instantiate the model
model = XiModel()
#best_loss = 0.0006978716119192541
best_loss = np.inf

df = {}
try:
    with open("../dataFiles/ICs.txt", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            df[float(row[0])] = float(row[1])
        print(df)
except FileNotFoundError:
    # Create the file if it doesn't exist
    with open("../dataFiles/ICs.txt", 'w') as f:
        print("File created successfully.")

closest_x = x_start
if x_start in df:
    best_loss = df[x_start]
else:
    closest_x = np.inf
    for i in df:
        if np.abs(i - x_start) < np.abs(closest_x - x_start):
            closest_x = i

# Load or instantiate the model
model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(closest_x)}_.pth"
new_model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(x_start)}_.pth"
print(model_path)
print(new_model_path)
if closest_x == x_start:
    assert(model_path == new_model_path)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded from file.")
else:
    print("No saved model found.")

print("closest_x =",closest_x)
print("best_loss =",best_loss)
ans = input("Proceed? (y/n): ")
if ans.lower() != 'y':
    exit()

# Function to compute the force (negative derivative of potential)
def force(x, xi):
#    return -(Omega**2 * (x - xi)) - (2.0 * A * (x - xi) / sigma) * torch.exp(-(x - xi)**2 / sigma)
    return -(Omega**2 * x) + (2 * A**3 * torch.sech(A * (x - xi))**2 * torch.tanh(A * (x - xi)))

# RK4 step for updating state
def rk4_step(x, v, xi_t, dt):
    def dxdt(v):
        return v

    def dvdt(x, xi):
        return force(x, xi) / m

    k1_x = dxdt(v)
    k1_v = dvdt(x, xi_t)

    k2_x = dxdt(v + 0.5 * dt * k1_v)
    k2_v = dvdt(x + 0.5 * dt * k1_x, xi_t)

    k3_x = dxdt(v + 0.5 * dt * k2_v)
    k3_v = dvdt(x + 0.5 * dt * k2_x, xi_t)

    k4_x = dxdt(v + dt * k3_v)
    k4_v = dvdt(x + dt * k3_x, xi_t)

    x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return x_new, v_new

# Define the xi function using the neural network
def xi(t):
    t_input = torch.tensor([[t]], dtype=torch.float32)  # Convert to tensor
    return model(t_input)[0, 0]  # Get the output from the model

# Loss function for optimization
def loss_func():
    MSE = 0.0
    global x_start, v_start
    x, v = x_start, v_start
    smoothness_penalty = 0.0  # Initialize smoothness penalty
    xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
    
    for t in t_values:
        xi_t = xi(t)  # Compute xi(t) at time t
        xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
        
        # Perform RK4 step
        x, v = rk4_step(x, v, xi_t, dt)
    print("Variance of xi =", np.var([i.detach().numpy() for i in xi_values_temp]))
    print(f"x = {x:.4f}, v = {v:.4f}, x_star = {x_star:.4f}, v_th = {v_th:.4f}, xi(T) = {xi_values_temp[-1]:.4f}")

    # Compute the smoothness penalty
    for i in range(1, len(xi_values_temp)):
        delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
        delta_t = t_values[i] - t_values[i - 1]
        
        # Compute the derivative
        if delta_t > 0:  # Avoid division by zero
            derivative = delta_xi / delta_t
            smoothness_penalty += torch.sum(derivative ** 2)  # Penalty based on the square of the derivative

    assert(smoothness_penalty > 0)
    # Regular MSE calculation
    MSE = (x_star - x) ** 2
    assert(MSE > 0)
    abs_v = abs(v)
    if abs_v > v_th:
        MSE += (abs_v - v_th) ** 2
    assert(MSE > 0)
    diff_xi_T = x_star - xi(T)
    MSE += (diff_xi_T ** 2)
    assert(MSE > 0)
    diff_xi_0 = xi(0)
    MSE += diff_xi_0 ** 2
    assert(MSE > 0)

    # Combine MSE with smoothness penalty (scale the penalty as needed)
    factor = 2.5e-7
    
#    > 3e-2 -> 2.5e-6
#    > 3e-3 -> 2.5e-7
#    > 3e-4 -> 2.5e-8
    total_loss = MSE + factor * smoothness_penalty  # Adjust the scale factor (0.1) to tune the smoothness constraint
    return total_loss

# Training loop
learning_rate = 0.01
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

plot_progress = False
epoch = 0

try:
    while True:
        optimizer.zero_grad()  # Zero the gradients
        loss_value = loss_func()
        loss_value.backward()  # Compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()  # Update weights
        
        print(f'Epoch {epoch}, Loss: {loss_value.item()}')

        # Check if the current loss is the best (lowest)
        if loss_value.item() < best_loss:
            best_loss = loss_value.item()  # Update best loss
            torch.save(model.state_dict(), new_model_path)  # Save the best model
            df[x_start] = best_loss
            # Write to CSV
            with open('../dataFiles/ICs.txt', 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in df.items():
                    writer.writerow([key, value])  # Write key-value pairs

            print(f"Best model saved with loss: {best_loss}")

        if plot_progress and epoch % 5 == 0:
            x_values, v_values, xi_values = [], [], []
            x, v = x_start, v_start  # Initial conditions
            for t in t_values:
                xi_t = xi(t)
                xi_values.append(xi_t.detach().numpy())
                x, v = rk4_step(x, v, xi_t, dt)
                x_values.append(x.detach().numpy())  # Use detach()
                v_values.append(v.detach().numpy())  # Use detach()
            plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
            plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
            plt.plot(t_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
            plt.legend()
            plt.draw()
            plt.pause(1)
            plt.close()
        epoch += 1

except KeyboardInterrupt:
    print("\nTraining interrupted. Saving data...")

    # Save the best model if it was updated
    if best_loss < float('inf'):  # Ensure that at least one model has been saved
        print(f"Best model was saved with loss: {best_loss}")
    else:
        print("No new best model found.")
        exit()
    
    model = XiModel()
    print("New model path loaded =",new_model_path)
    model.load_state_dict(torch.load(new_model_path, weights_only=True))
    xi = lambda t: model(torch.tensor([[t]], dtype=torch.float32))[0, 0] # Get the output from the model
    # Save data to CSV
    data_path = "../dataFiles/trajectory_data.csv"
    x_values, v_values, xi_values = [], [], []
    x, v = x_start, v_start # Initial conditions
    print(f"x_0, v_0 = {x}, {v}")
    for t in t_values:
        xi_t = xi(t)
        xi_values.append(xi_t.detach().numpy())
        x, v = rk4_step(x, v, xi_t, dt)
        x_values.append(x.detach().numpy())
        v_values.append(v.detach().numpy())
    with open(data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
        for i in range(len(t_values)):
            writer.writerow([t_values[i], xi_values[i].item(), x_values[i].item(), v_values[i].item()])
    print("Data saved to CSV.")
    
    plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
    plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
    plt.plot(t_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha = 0.2)  # Red dashed line at y = 0.5
    plt.axhline(y=0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.01
    plt.axhline(y=-0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = -0.01

    plt.xlabel('t')
    plt.title(f'$x_0$ = {x_start}')
    plt.legend()
    plt.savefig("trajectory_data.svg")
    system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(x_start)}_.pdf trajectory_data.svg")
    system("rm trajectory_data.svg")
    system(f"open trajectory_data_IC_{flt_to_str(x_start)}_.pdf")

