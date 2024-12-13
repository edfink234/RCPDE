import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tanh, arccos, sqrt, log as ln, arcsin, log, arcsin as asin, arccos as acos, exp
from matplotlib import rcParams
import torch
import torch.nn as nn
import torch.optim as optim
import os
# Constants for the potential
m = 1.0        # Mass
Omega = 1.0    # Frequency of the harmonic trap
A = 1.0        # Amplitude of the Gaussian potential
sigma = 1.0    # Width of the Gaussian potential
T = 10.0       # Final time
dt = 0.01      # Time step
x_star = 0.5   # Final position sought
v_th = 0.01    # Velocity threshold
sech = lambda x: 1/torch.cosh(x)
torch.sech = sech

class XiModel(nn.Module):
    def __init__(self):
        super(XiModel, self).__init__()
        # Initialize layers with reduced neurons
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 4)
        self.fc3 = nn.Linear(4, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc_out = nn.Linear(4, 1)
        self.fc_skip = nn.Linear(1, 1)
        
    def forward(self, inputs):
        x = torch.tanh(self.fc1(inputs))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output_main = self.fc_out(x)
        output_skip = self.fc_skip(inputs)
        return output_main + output_skip

def flt_to_str(flt):
    return str(flt).replace(".","_point_")

# Initial conditions
x = 1.0  # Initial position x(0)
v = 0.0  # Initial velocity v(0)

# Instantiate the model
model = XiModel()

# Load or instantiate the model
model_path = f"xi_model_IC_{flt_to_str(float(x))}_.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    print("Model loaded from file.")
else:
    print("No saved model found. Starting from scratch.")

# Define the xi function using the neural network
def xi(t):
    t_input = torch.tensor([[t]], dtype=torch.float32)  # Convert to tensor
    return model(t_input)[0, 0]  # Get the output from the model

def potential(x, xi):
    V_MT = 0.5 * Omega * Omega * x * x
    temp = torch.sech(A * (x - xi))
    V_SECH = A * A * temp * temp
    return V_MT + V_SECH

# Function to compute the force (negative derivative of potential)
def force(x, xi):
    return -(Omega**2 * x) + (2 * A**3 * torch.sech(A * (x - xi))**2 * torch.tanh(A * (x - xi)))

# RK4 step for updating state
def rk4_step(x, v, xi_t, dt):
    # Derivatives for RK4 method
    def dxdt(v):
        return v
    def dvdt(x, xi):
        return force(x, xi) / m

    # Compute RK4 coefficients
    k1_x = dxdt(v)
    k1_v = dvdt(x, xi_t)
    
    k2_x = dxdt(v + 0.5 * dt * k1_v)
    k2_v = dvdt(x + 0.5 * dt * k1_x, xi_t)
    
    k3_x = dxdt(v + 0.5 * dt * k2_v)
    k3_v = dvdt(x + 0.5 * dt * k2_x, xi_t)
    
    k4_x = dxdt(v + dt * k3_v)
    k4_v = dvdt(x + dt * k3_x, xi_t)

    # Update x and v
    x_new = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    v_new = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return x_new, v_new

# Simulation
x_values = []   # To store x(t)
v_values = []   # To store v(t)
xi_values = []  # To store xi(t)
t_values = np.linspace(1e-8, T, int(T/dt))
print(f"Number of t values = {len(t_values)}")

# Time evolution loop
for t in t_values:
    xi_t = xi(t)  # Compute xi(t) at time t
    xi_values.append(xi_t.detach().numpy())
    
    # Perform RK4 step
    x, v = rk4_step(x, v, xi_t, dt)
    
    # Store the position x(t) and velocity v(t)
    x_values.append(x.detach().numpy())
    v_values.append(v.detach().numpy())

# Plot the results
#point_sz = 0.1 * (rcParams['lines.markersize'] ** 2)
plt.plot(t_values, x_values, label='x(t) [m]', color='blue', linestyle='solid')
plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
plt.plot(t_values, xi_values, label = r'$\xi(t)$ [m]', color='red', linestyle='dashed')

xi_f = xi_values[-1]

# Define the oscillatory xi function for post T=10
def xi_oscillating(t):
    return 0.25 * np.sin(2 * np.pi * (t - 10) / 32) + xi_f  # Adjust phase to start oscillating at T=10

t_values = np.linspace(T+dt, 80*T, int(T/dt))
x_values_osc = []   # To store x(t)
v_values_osc = []   # To store v(t)
xi_values_osc = []  # To store xi(t)

for t in t_values:
    xi_t = xi_oscillating(t)  # Compute xi(t) at time t
    xi_values_osc.append(xi_t)
    
    # Perform RK4 step
    x, v = rk4_step(x, v, xi_t, dt)
    
    # Store the position x(t) and velocity v(t)
    x_values_osc.append(x.detach().numpy())
    v_values_osc.append(v.detach().numpy())

plt.plot(t_values, x_values_osc, label='x(t) osc [m]', color='purple', linestyle = 'solid')
plt.plot(t_values, v_values_osc, label='v(t) osc [m/s]', color='cyan', linestyle='-.')
plt.plot(t_values, xi_values_osc, label = r'$\xi(t)$ osc [m]', color='orange', linestyle='--')

plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
plt.savefig(f"ControlThenLetRun_IC_{flt_to_str(float(x))}_.png", dpi=5*96)
os.system(f"open ControlThenLetRun_IC_{flt_to_str(float(x))}_.png")
