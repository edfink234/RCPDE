import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import csv
import os
from os import system
import csv
import matplotlib.animation as animation
from random import choice
from time import time

#Setting the random seeds!!!
np.random.seed(42)
torch.manual_seed(42)
load_model = True

sech = lambda x: 1/torch.cosh(x)
torch.sech = sech

def flt_to_str(flt):
    return str(flt).replace(".","_point_")

# Constants for the potential
m = 1.0        # Mass
Omega = 0.2    # Frequency of the harmonic trap
A = 1.0        # Amplitude of the Gaussian potential
sigma = 1.0    # Width of the Gaussian potential
T = 10.0       # Final time
dt = 0.01      # Time step
x_star = 0.0   # Final position sought
v_th = 0.01    # Velocity threshold
x_th = 0.01    # Position threshold
to_time = {"timed":True, "time": 3600}
criterion = lambda: True if not to_time["timed"] else time() - start_time < to_time["time"]
automate = True

smoothness_penalty_factor = 1e-5 # penalty for lack of smoothness
time_penalty_factor = 1e-5 #penalty for taking longer

t_values = np.linspace(1e-8, T, int(T / dt))
delta_t = t_values[1] - t_values[0]
t_test_values = np.linspace(1e-8, T, int(T / dt))
with open("temp.txt", "r") as f:
    x_start, v_start = float(f.read()), 0.0 #IC
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
best_loss = np.inf
best_t_value = T

df = {}
try:
    with open("../dataFiles/ICs.txt", 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            df[float(row[0])] = (float(row[1]), float(row[2]))
        print(df)
except FileNotFoundError:
    # Create the file if it doesn't exist
    with open("../dataFiles/ICs.txt", 'w') as f:
        print("File created successfully.")

closest_x = x_start
if x_start in df:
    best_loss, best_t_value = df[x_start]
    t_test_values = np.linspace(1e-8, best_t_value, int(best_t_value / dt))
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
if os.path.exists(model_path) and load_model:
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print("Model loaded from file.")
elif os.path.exists(model_path) and not load_model:
    print("Model exists but not loaded.")
else:
    print("No saved model found.")

print("closest_x =",closest_x)
print("best_loss =",best_loss)
if not automate:
    ans = input("Proceed? (y/n): ")
    if ans.lower() != 'y':
        exit()

def potential(x, xi):
    V_MT = 0.5 * Omega * Omega * x * x
    temp = torch.sech(A * (x - xi))
    V_SECH = A * A * temp * temp
    return V_MT + V_SECH

# Function to compute the force (negative derivative of potential)
def force(x, xi):
    return -(Omega**2 * x) + (2 * A**3 * torch.sech(A * (x - xi))**2 * torch.tanh(A * (x - xi)))

def dxdt(v):
    return v

def dvdt(x, xi):
    return force(x, xi) / m

# RK4 step for updating state
def rk4_step(x, v, xi_t, dt):
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

'''
What I really need is a function that scans the entire array of time values, computes the loss-function at the time, gets the total loss, and then if that is better than the best loss achieved (in the array of time values it's scanning) then the best loss and corresponding best time are updated. 
But of course, I want to encourage better solutions earlier on, so I'll multiply the best loss by some value that linearly increases with the time
'''

def loss_func():
    global x_start, v_start, m, delta_t, x_th, v_th, smoothness_penalty_factor, time_penalty_factor
    x, v = x_start, v_start
#    a = force(torch.tensor(x_start), xi(0.0)) / m
    smoothness_penalty = 0.0  # Initialize smoothness penalty
    xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
    best_loss = np.inf
    best_time = np.inf
    best_time_idx = np.inf
    
    for i, t in enumerate(t_values):
        xi_t = xi(t)  # Compute xi(t) at time t
        xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
        x, v = rk4_step(x, v, xi_t, dt)
        
        x_star_x_diff = (x_star - x)
        x_star_xi_diff = (x_star - xi(t))
        xi_0 = xi(0)
        
        if i > 1:
            MSE = (x_star_x_diff*x_star_x_diff) + (v*v) + (x_star_xi_diff*x_star_xi_diff) + (xi_0*xi_0)
            if MSE < best_loss:
                best_time = t
                best_loss = MSE
                best_time_idx = i
                
        
    for i in range(1, best_time_idx+1):
        delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
        derivative = delta_xi / delta_t
        smoothness_penalty += torch.sum(derivative ** 2)  # Penalty based on the square of the "derivative"
    smoothness_penalty /= best_time_idx
    
    return (best_loss + smoothness_penalty_factor*smoothness_penalty + time_penalty_factor*best_time), best_time

# Loss function for optimization

def closure():
    optimizer.zero_grad()  # Clear previous gradients
    loss, _ = loss_func()  # Compute the loss
    loss.backward()  # Backpropagate
    return loss

# Define Newton's method function
def newton_method():
    global best_loss, best_t_value, model, new_model_path, df, x_start, t_test_values, criterion
    # Extract model parameters
    current_params = {name: param.clone() for name, param in model.named_parameters()}
    
    # Compute initial loss
    best_params = {name: param.clone() for name, param in current_params.items()}
    
    while criterion():
        # Compute gradient and Hessian for each parameter
        grads = {}
        hessians = {}

        # Compute loss and gradients
        loss, _ = loss_func()
        grad_tensors = torch.autograd.grad(
            loss,
            model.parameters(),
            create_graph=True,
            retain_graph=True
        )

        for (name, param), grad in zip(model.named_parameters(), grad_tensors):
            grads[name] = grad
            # Compute Hessian (second derivatives)
            hessian = []
            for g in grad.view(-1):  # Flatten gradient for Hessian computation
                hessian.append(
                    torch.autograd.grad(g, param, retain_graph=True)[0].view(-1)
                )
            hessians[name] = torch.stack(hessian).view(param.shape + param.shape)

        # Compute Newton update for each parameter
        new_params = {}
        for name, param in current_params.items():
            grad = grads[name].view(-1)  # Flatten gradient
            hessian = hessians[name].view(grad.numel(), grad.numel())  # Reshape Hessian to match flattened gradient
            
            # Damped Hessian to ensure positive definiteness
            hessian_damped = hessian + 1e-4 * torch.eye(hessian.size(0), device=hessian.device)

            # Solve H Δx = -grad
            try:
                update = torch.linalg.solve(hessian_damped, -grad)  # Solve for flattened update
            except RuntimeError as e:
                print(f"Hessian inversion failed for parameter {name}: {e}")
                update = -grad  # Fallback to gradient descent step
            
            # Reshape the update to match parameter shape
            new_params[name] = param + learning_rate * update.view(param.shape)


        # Update model parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(new_params[name])

        # Compute new loss
        new_loss, new_time = loss_func()

        # Check for improvement
        if new_loss < best_loss:
            # Update best parameters
            best_loss = new_loss
            best_t_value = new_time
            best_params = {name: param.clone() for name, param in current_params.items()}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(best_params[name])
            # Save best model
            torch.save(model.state_dict(), new_model_path)
            df[x_start] = (best_loss.detach().numpy(), best_t_value)
            if best_t_value != t_test_values[-1]:
                print(f"New best t value = {best_t_value}")
                t_test_values = np.linspace(1e-8, best_t_value, best_t_value / dt)
            # Write to CSV
            with open('../dataFiles/ICs.txt', 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in df.items():
                    writer.writerow([key, *value])  # Write key-value pairs
            print(f"Best model saved with loss: {best_loss}")
        else:
            # Revert to previous parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(current_params[name])

        # Log current loss and time
        print(f"Curr Loss = {new_loss}, Curr Time = {new_time:.6f}")

    # Restore best parameters to the model
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(best_params[name])

# Define brute force function
def brute_force(fine = False):
    global best_loss, best_t_value, model, new_model_path, df, x_start, t_test_values
    # Extract model parameters
    current_params = {name: param.clone() for name, param in model.named_parameters()}
    initial_temp = 1
    temperature = initial_temp
    cooling_rate=1
    
    # Compute initial loss
    best_params = {name: param.clone() for name, param in current_params.items()}

    while criterion():
        # Generate a random perturbation
        perturbed_params = {}
        if fine:
            name, param = choice(tuple(current_params.items()))
            perturbed_params = {
                key: (torch.randn_like(param) * temperature + param if key == name else param)
                for key, param in current_params.items()
            }
        else:
            perturbed_params = {
                name: torch.randn_like(param) * (temperature) + param
                for name, param in current_params.items()
            }
        # Update model with perturbed parameters
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(perturbed_params[name])

        # Compute new loss
        new_loss, new_time = loss_func()

        # Compute change in loss
        delta_loss = new_loss - best_loss
        

        # Metropolis criterion
        if delta_loss < 0:
            # Accept new parameters
            current_params = perturbed_params
            best_loss = new_loss
            best_t_value = new_time
            best_params = {name: param.clone() for name, param in current_params.items()}
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(best_params[name])
                            
                            
            torch.save(model.state_dict(), new_model_path)  # Save the best model
            df[x_start] = (best_loss.detach().numpy(), best_t_value)
            if best_t_value != t_test_values[-1]:
                print(f"New best t value = {best_t_value}")
                t_test_values = np.linspace(1e-8, best_t_value, best_t_value/dt)
            # Write to CSV
            with open('../dataFiles/ICs.txt', 'w', newline='') as file:
                writer = csv.writer(file)
                for key, value in df.items():
                    writer.writerow([key, *value])  # Write key-value pairs
                print(f"Best model saved with loss: {best_loss}")
        elif torch.exp(-delta_loss / temperature) > np.random.random() and not fine:
            current_params = perturbed_params
        else:
            # Revert to previous parameters
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(current_params[name])
        
        temperature *= cooling_rate
        print(f"Curr Loss = {new_loss:.6f}, Curr Time = {new_time:.6f}")


    # Restore best parameters to the model
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(best_params[name])

# Usage Example
# Initialize neural network
xi_model = XiModel()

# Loss function for optimization
#def loss_func():
#    MSE = 0.0
#    global x_start, v_start, m, delta_t, x_th, v_th
#    x, v = x_start, v_start
##    a = force(torch.tensor(x_start), xi(0.0)) / m
#    smoothness_penalty = 0.0  # Initialize smoothness penalty
#    xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
#    time_loss = t_values[-1]
#    time_end = 0
#    MSE = 0
#    t_less_than_T = (xi(0) < x_th)
#    t_less_than_T_points_best = 0
#    
#    for t in t_values:
#        xi_t = xi(t)  # Compute xi(t) at time t
#        xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
#        curr_points = 0
#        # Perform RK4 step
#        x, v = rk4_step(x, v, xi_t, dt)
#        if abs(x - x_star) < x_th:
#            curr_points += 1
#        if abs(v) < v_th:
#            curr_points += 1
#        if abs(x_star - xi(t)) < x_th:
#            curr_points += 1
#        if curr_points > t_less_than_T_points_best:
#            t_less_than_T_points_best = curr_points
#            if curr_points == 3:
#                if t_less_than_T:
#                    time_loss = t
#                break
#                
##    print("Variance of xi =", np.var([i.detach().numpy() for i in xi_values_temp]))
#    print(f"x = {x:.4f}, v = {v:.4f}, x_star = {x_star:.4f}, v_th = {v_th:.4f}, xi(T) = {xi_values_temp[-1]:.4f}")
#
#    t_less_than_T_points_best += t_less_than_T
#    time_end = time_loss
#    if time_loss == t_values[-1]:
#        time_loss *= (10 - 2.25*t_less_than_T_points_best)
#        
#    # Compute the smoothness penalty
#    for i in range(1, len(xi_values_temp)):
#        delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
#        derivative = delta_xi / delta_t
#        smoothness_penalty += torch.sum(derivative ** 2)  # Penalty based on the square of the "derivative"
#
#    assert(smoothness_penalty > 0)
#    # Regular MSE calculation
#    MSE = (x_star - x) ** 2
#    assert(MSE > 0)
#    abs_v = abs(v)
#    if abs_v > v_th:
#        MSE += (abs_v - v_th) ** 2
#    assert(MSE > 0)
#    diff_xi_T = x_star - xi(time_end)
#    MSE += (diff_xi_T ** 2)
#    assert(MSE > 0)
#    diff_xi_0 = xi(0)
#    MSE += diff_xi_0 ** 2
#    assert(MSE > 0)
#    MSE += time_loss
#
#    # Combine MSE with smoothness penalty (scale the penalty as needed)
#    factor = 2.5e-7
#    
##    > 3e-2 -> 2.5e-6
##    > 3e-3 -> 2.5e-7
##    > 3e-4 -> 2.5e-8
#    total_loss = MSE + factor * smoothness_penalty  # Adjust the smoothness_penalty to tune the smoothness constraint
#    return total_loss, time_end

# Training loop
learning_rate = 0.1
Algorithm = "adam"
fine = True #for brute force
#optimizer = optim.SGD(model.parameters(), lr=learning_rate)
if Algorithm == "lbfgs":
    optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
elif Algorithm == "adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
elif Algorithm == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
elif Algorithm == "adamax":
    optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
elif Algorithm == "adafactor":
    optimizer = optim.Adafactor(model.parameters(), lr=learning_rate)
elif Algorithm == "adamw":
    optimizer = optim.Adafactor(model.parameters(), lr=learning_rate)
elif Algorithm == "asgd":
    optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
elif Algorithm == "sparse adam":
    optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)

plot_progress = False
epoch = 0
start_time = time()

try:
    if Algorithm == "brute force":
        # Define constants and call the simulated annealing function
        brute_force(fine)
    elif Algorithm == "newton":
        newton_method()
    else:
        while criterion():
            optimizer.zero_grad()  # Zero the gradients
            loss_value, t_value = loss_func()
            loss_value.backward()  # Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    #        optimizer.step()  # Update weights
            if Algorithm == "lbfgs":
                optimizer.step(closure)
            else:
                optimizer.step()
            
            print(f'Epoch {epoch}, Loss: {loss_value.item()}')

            # Check if the current loss is the best (lowest)
            if loss_value.item() < best_loss:
                best_loss = loss_value.item()  #Update best loss
                best_t_value = t_value #Update corresponding best time
                torch.save(model.state_dict(), new_model_path)  # Save the best model
                df[x_start] = (best_loss, best_t_value)
                if best_t_value != t_test_values[-1]:
                    print(f"New best t value = {best_t_value}")
                    t_test_values = np.linspace(1e-8, best_t_value, int(best_t_value/dt))
                # Write to CSV
                with open('../dataFiles/ICs.txt', 'w', newline='') as file:
                    writer = csv.writer(file)
                    for key, value in df.items():
                        writer.writerow([key, *value])  # Write key-value pairs

                print(f"Best model saved with loss: {best_loss}")

            if plot_progress and epoch % 5 == 0:
                x_values, v_values, xi_values = [], [], []
                x, v = x_start, v_start  # Initial conditions
                for t in t_test_values:
                    xi_t = xi(t)
                    xi_values.append(xi_t.detach().numpy())
                    x, v = rk4_step(x, v, xi_t, dt)
                    a = force(x, xi_t) / m
                    x_values.append(x.detach().numpy())  # Use detach()
                    v_values.append(v.detach().numpy())  # Use detach()
                    a_values.append(a.detach().numpy())
                plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
                plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
                plt.plot(t_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
                plt.plot(t_values, a_values, label='a(t) [m/$s^2$]', color='purple', linestyle='-.')

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
    x_values, v_values, a_values, xi_values = [], [], [], []
    x, v = x_start, v_start # Initial conditions
    a = force(torch.tensor(x_start), xi(0.0)) / m
    print(f"x_0, v_0 = {x}, {v}")
    for t in t_test_values:
        xi_t = xi(t)
        xi_values.append(xi_t.detach().numpy())
        x, v = rk4_step(x, v, xi_t, dt)
        a = force(x, xi_t) / m
        x_values.append(x.detach().numpy())
        v_values.append(v.detach().numpy())
        a_values.append(a.detach().numpy())
        assert(len(a_values) == len(x_values))
    with open(data_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
        for i in range(len(t_test_values)):
            writer.writerow([t_values[i], xi_values[i].item(), x_values[i].item(), v_values[i].item()])
    print("Data saved to CSV.")
    
    plt.plot(t_test_values, x_values, label='x(t) [m]', color='blue')
    plt.plot(t_test_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
    plt.plot(t_test_values, a_values, label='a(t) [m/$s^2$]', color='purple', linestyle='-.')
    plt.plot(t_test_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
#    plt.axhline(y=0.5, color='black', linestyle='--', alpha = 0.2)  # Red dashed line at y = 0.5
#    plt.axhline(y=0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.01
#    plt.axhline(y=-0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = -0.01
    plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.0

    plt.xlabel('t')
    plt.title(f'$x_0$ = {x_start}')
    plt.legend()
    plt.savefig("trajectory_data.svg")
    system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(x_start)}_.pdf trajectory_data.svg")
    system("rm trajectory_data.svg")
    system(f"open trajectory_data_IC_{flt_to_str(x_start)}_.pdf")
    system(f"cp trajectory_data_IC_{flt_to_str(x_start)}_.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
    system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.png")
    system(f"open ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.png")
    
    if not automate:
        answer = input("Movie (y/n)? ")
        if not answer.lower().startswith('y'):
            print("Movie creation skipped.")
            exit()
        
    N = len(t_test_values)
    x_range = np.linspace(-10, 10, N)
    fig, ax = plt.subplots(figsize=(8, 6))
    # Initialize plot elements
    dot, = ax.plot([], [], 'ro', markersize=8, label = "Particle")
    curve, = ax.plot([], [], 'b-', lw=2)
    gold_dot, = ax.plot([], [], 'yo', markersize=8, label = "$x^*$")

    # Set plot limits and labels
    ax.set_xlim(x_range[0], x_range[-1])
    ax.set_ylim(0, 3)
    ax.set_xlabel("x")
    ax.set_ylabel("Potential")
    ax.legend()
    ax.set_title(f"Particle Movement and Potential Curve for $x_0$ = {x_start}")
    sech = lambda x: 1/np.cosh(x)
    
    def potential(x, xi):
        V_MT = 0.5 * Omega * Omega * x * x
        temp = sech(A * (x - xi))
        V_SECH = A * A * temp * temp
        return V_MT + V_SECH

    # Initialization function
    def init():
        gold_dot.set_data([], [])
        dot.set_data([], [])
        curve.set_data([], [])
        return dot, curve, gold_dot

        # Update function
    def update(i):
        t = (i) * (best_t_value / (N - 1))  # Calculate current time
        y_values = potential(x_range, xi_values[i])
        dot.set_data([x_values[i]], [potential(x_values[i], xi_values[i])])
        curve.set_data(x_range, y_values)
        gold_dot.set_data([x_star], [potential(x_star, xi_values[i])])
        ax.set_title(f"Particle Movement and Potential Curve for $x_0$ = {x_values[0]:.1f}, t = {t:.2f}")
        return dot, curve, gold_dot

    # Create animation
    fps = N/best_t_value

    ani = animation.FuncAnimation(
        fig, update, frames=N, init_func=init, blit=True, interval = 1000/fps
    )
    
    # Save the animation
    ani.save(f"../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.mp4", writer=animation.FFMpegWriter(fps=2*fps))
    system(f"open ../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.mp4")
    print(f"Movie saved as '../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_.mp4'")
            
