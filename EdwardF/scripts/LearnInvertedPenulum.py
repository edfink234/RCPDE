import numpy as np
import pandas as pd
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
from scipy.optimize import fsolve
from warnings import filterwarnings
filterwarnings('ignore')

def master_func(m = 1.0, Omega = 0.2, A = 1.0, b = 1.0):
    #Setting the random seeds!!!
    np.random.seed(42)
    torch.manual_seed(42)
    load_model = True

    sech = lambda x: 1/torch.cosh(x)
    torch.sech = sech

    def flt_to_str(flt):
        return str(flt).replace(".","_point_")

    def add_or_update_row(df, new_row_data):
        """
        Adds a new row to the DataFrame if no match is found 
        for 'x_0', 'A', 'b', 'm', 'Omega'. 
        If a match exists, overwrites the existing row.

        Args:
            df: The DataFrame to modify.
            new_row_data: A dictionary containing the data for the new row.

        Returns:
            The modified DataFrame.
        """

        # Create a temporary DataFrame for efficient matching
        temp_df = df[['x_0', 'A', 'b', 'm', 'Omega']]

        # Find rows with matching parameters
        matching_rows = temp_df[
            (temp_df['x_0'] == new_row_data['x_0']) &
            (temp_df['A'] == new_row_data['A']) &
            (temp_df['b'] == new_row_data['b']) &
            (temp_df['m'] == new_row_data['m']) &
            (temp_df['Omega'] == new_row_data['Omega'])
        ]

        if not matching_rows.empty:
            # Overwrite existing row
            row_index = matching_rows.index[0]  # Get the index of the first (and only) match
            df.loc[row_index] = new_row_data
        else:
            # Add a new row
            df.loc[len(df)] = new_row_data

        return df

    #NOTE: Roberto has periodic boundary conditions in his PDE.
    # Constants for the potential
#    m = 1.0        # Mass
#    Omega = 0.2    # Frequency of the harmonic trap
#    A = 1.2        # Amplitude of the potential
#    b = 1.0        # Width of the potential
    sigma = 1.0    # Width of the (Gaussian) potential, not used currently
    T = 10.0       # Final time
    dt = 0.01      # Time step
    x_star = 0.0   # Final position sought
    v_th = 0.01    # Velocity threshold, not used currently
    x_th = 0.01    # Position threshold, not used currently
    to_time = {"timed": False, "time": 3600}
    to_loss = {"loss thresholded": True, "threshold": 1.3e-2}
    raiseBaseException = True
    def criterion():
#        global to_time, to_loss, best_loss
        if not to_time["timed"] and to_loss["loss thresholded"]:
            return best_loss >= to_loss['threshold']
        elif to_time["timed"] and not to_loss["loss thresholded"]:
            return time() - start_time < to_time["time"]
        elif to_time["timed"] and to_loss["loss thresholded"]:
            return (time() - start_time < to_time["time"]) or (best_loss >= to_loss['threshold'])
        else:
            return True
            
    def potential(x, xi):
        V_MT = 0.5 * Omega * Omega * x * x
        temp = torch.sech(A * (x - xi))
        V_SECH = A * A * temp * temp
        return V_MT + V_SECH

    def potential(x, xi):
        V_MT = 0.5 * Omega * Omega * x * x
        temp = torch.sech(b * (x - xi))
        V_SECH = A * temp * temp
        return V_MT + V_SECH

    # Function to compute the force (negative derivative of potential)
    def force(x, xi):
        return -(Omega**2 * x) + (2 * A**3 * torch.sech(A * (x - xi))**2 * torch.tanh(A * (x - xi)))

    def force(x, xi):
        return -(Omega**2 * x) + (2 * A*b * torch.sech(b * (x - xi))**2 * torch.tanh(b * (x - xi)))

    def min_force(x):
        return (Omega**2 * x) - (2 * A*b * np.cosh(b * (x))**(-2) * np.tanh(b * (x)))

    #criterion = lambda: True if not to_time["timed"] else time() - start_time < to_time["time"]
    automate = True
    produceInverse = False
    saveLibTorch = True
    useLibTorch = True

    smoothness_penalty_factor = 1e-3 # penalty for lack of smoothness of xi
    time_penalty_factor = 1e-3 #penalty for taking longer
    velocity_penalty = 1e-3 #penalty for max(abs(v))
    xi_penalty = 1e-3 #penalty for max(abs(xi))

    t_values = np.linspace(1e-8, T, int(T / dt))
    delta_t = t_values[1] - t_values[0]
    t_test_values = np.linspace(1e-8, T, int(T / dt))


    # Initial guess for the root (you might need to adjust this)
    x0 = 1.0

    # Find the root
    root, info, ier, mesg = fsolve(min_force, x0, full_output=True)

    if ier == 1:
        print(f"Root found: {root[0]}")
        x_start, v_start = float(root[0]), 0.0
    else:
        print(f"Root finding failed: {mesg}")
        exit()

    x_start = round(float(x_start), 4)

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
    example = torch.tensor([[0.5]], dtype=torch.float32)
    best_loss = np.inf
    best_t_value = T

    df = None
    try:
        df = pd.read_csv("../dataFiles/ICs.txt", names=("x_0", "A", "b", "m", "Omega", "best_loss", "best_time"))
    except FileNotFoundError:
        # Create the file if it doesn't exist
        with open("../dataFiles/ICs.txt", 'w') as f:
            print("File created successfully.")

    parameters = df[(df['x_0']==x_start) & (df['A']==A) & (df['b']==b) & (df['m']==m) & (df['Omega']==Omega)]
    closest_row_idx = -1
    closest_x, closest_A, closest_b, closest_m, closest_Omega = x_start, A, b, m, Omega

    if len(parameters):
        print(parameters)
        best_loss, best_t_value = parameters['best_loss'].item(), parameters['best_time'].item()
        try:
            t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
        except:
            t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
        closest_row_idx = parameters.index[0]
    else:
        #Extract the row with the closest 'x_0', 'A', 'b', 'm', 'Omega' to (x_start, A, b, m, Omega) based on euclidean distance
        # Create a new DataFrame with the relevant columns for distance calculation
        relevant_cols = ['x_0', 'A', 'b', 'm', 'Omega']
        temp_df = df[relevant_cols]

        # Calculate Euclidean distance between each row and the target values
        temp_df['distance'] = np.linalg.norm(temp_df - [x_start, A, b, m, Omega], axis=1)
        # Find the index of the row with the minimum distance
        closest_row_idx = temp_df['distance'].idxmin()
        parameters = temp_df.iloc[closest_row_idx]
        closest_x, closest_A, closest_b, closest_m, closest_Omega = round(parameters['x_0'], 4), round(parameters['A'], 2), round(parameters['b'], 2), round(parameters['m'], 2), round(parameters['Omega'], 2)

    # Load or instantiate the model
    model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(closest_x)}_{flt_to_str(round(closest_A, 2))}_{flt_to_str(round(closest_b, 2))}_{flt_to_str(round(closest_m, 2))}_{flt_to_str(round(closest_Omega, 2))}_.pth"
    new_model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pth"
    print(model_path)
    print(new_model_path)
    if closest_x == x_start and closest_A == A and closest_b == b and closest_m == m and closest_Omega == Omega:
        print("exact match found")
        assert(model_path == new_model_path)
    else:
        print(f"closest_x = {closest_x}, closest_A = {closest_A}, closest_b = {closest_b}, closest_m = {closest_m}, closest_Omega = {closest_Omega}")
        print(f"x_start = {x_start}, A = {A}, b = {b}, m = {m}, Omega = {Omega}")
        print("problem!")
        exit()
    if os.path.exists(model_path) and load_model:
        if useLibTorch:
            model = torch.jit.load(model_path.replace(".pth", ".pt"))
            new_model_path = new_model_path.replace(".pth", ".pt")
        else:
            model.load_state_dict(torch.load(model_path))#, weights_only=True))
        print("Model loaded from file.")
        print(f"Example input of {example} yields: {model(example)[0, 0]}")
    elif os.path.exists(model_path) and not load_model:
        print("Model exists but not loaded.")
    else:
        print("No saved model found.")

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

    def loss_func():
#        global x_start, v_start, m, delta_t, x_th, v_th, smoothness_penalty_factor, time_penalty_factor, velocity_penalty
        x, v = x_start, v_start
    #    a = force(torch.tensor(x_start), xi(0.0)) / m
        smoothness_penalty = 0.0  # Initialize smoothness penalty
        xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
        v_values_temp = [] # Temporary storage for v values to calculate max velocity
        best_loss_ = np.inf
        best_time = np.inf
        best_time_idx = np.inf
        
        for i, t in enumerate(t_values):
            xi_t = xi(t)  # Compute xi(t) at time t
            xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
            x, v = rk4_step(x, v, xi_t, dt)
            v_values_temp.append(abs(v))
            
    #        print(f"i = {i}, x = {x}, v = {v}")
    #        print(f"t = {t}, xi(t) = {xi_values_temp[-1]}\n")
                        
            x_star_x_diff = (x_star - x)
            x_star_xi_diff = (x_star - xi(t))
            xi_0 = xi(0)
            
            if i > 1:
                MSE = (x_star_x_diff*x_star_x_diff) + (v*v) + (x_star_xi_diff*x_star_xi_diff) + (xi_0*xi_0)
                if MSE < best_loss_:
    #                print(f"x_star_x_diff*x_star_x_diff = {x_star_x_diff*x_star_x_diff}")
    #                print(f"v*v = {v*v}")
    #                print(f"x_star_xi_diff*x_star_xi_diff = {x_star_xi_diff*x_star_xi_diff}")
    #                print(f"xi_0*xi_0 = {xi_0*xi_0}")
                    best_time = t
                    best_loss_ = MSE
                    best_time_idx = i
                    
        v_best = v_values_temp[0]
        xi_best = abs(xi_values_temp[0])
        for i in range(1, int(best_time_idx)+1):
            delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
            derivative = delta_xi / delta_t
            smoothness_penalty += derivative*derivative#torch.sum(derivative ** 2)  # Penalty based on the square of the "derivative"
            if v_values_temp[i] > v_best:
                v_best = v_values_temp[i]
            abs_xi_temp_i = abs(xi_values_temp[i])
            if abs_xi_temp_i > xi_best:
                xi_best = abs_xi_temp_i
                
        smoothness_penalty /= best_time_idx
        
    #    print(f"best_loss_ = {best_loss_}, smoothness_penalty = {smoothness_penalty},\nbest_time = {best_time}, v_best = {v_best:}\nabs_xi_temp_i = {xi_best}")
    #    exit()
        return (best_loss_ + smoothness_penalty_factor*smoothness_penalty + time_penalty_factor*best_time + velocity_penalty*v_best + xi_penalty*xi_best), best_time

    # Loss function for optimization

    def closure():
        optimizer.zero_grad()  # Clear previous gradients
        loss, _ = loss_func()  # Compute the loss
        loss.backward()  # Backpropagate
        return loss

    # Define Newton's method function
    def newton_method():
#        global best_loss, best_t_value, model, new_model_path, df, x_start, t_test_values, criterion, saveLibTorch
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
                torch.save(model.state_dict(), new_model_path if new_model_path.endswith(".pth") else new_model_path+"h")  # Save the best model
                if saveLibTorch:
                    traced_script_module = torch.jit.trace(model, example)
                    traced_script_module.save(new_model_path.replace(".pth",".pt"))
                    print("libtorch version saved")
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A,2), 'b': round(b,2), 'm': round(m,2), 'Omega': round(Omega,2), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    print(f"New best t value = {best_t_value}")
                    t_test_values = np.linspace(1e-8, best_t_value, best_t_value / dt)
                # Write to CSV
#                df.to_csv('../dataFiles/ICs.txt', header=None, index=False) #TODO: uncomment
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
    def brute_force(fine = False, coolingRate = 0.99, anneal = False, initial_temp = 1):
#        global best_loss, best_t_value, model, new_model_path, df, x_start, t_test_values
        # Extract model parameters
        current_params = {name: param.clone() for name, param in model.named_parameters()}
        temperature = initial_temp
        cooling_rate=coolingRate
        
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
           
                torch.save(model.state_dict(), new_model_path if new_model_path.endswith(".pth") else new_model_path+"h")  # Save the best model
                if saveLibTorch:
                    traced_script_module = torch.jit.trace(model, example)
                    traced_script_module.save(new_model_path.replace(".pth",".pt"))
                    print("libtorch version saved")
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A,2), 'b': round(b,2), 'm': round(m,2), 'Omega': round(Omega,2), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    print(f"New best t value = {best_t_value}")
                    t_test_values = np.linspace(1e-8, best_t_value, best_t_value / dt)
                # Write to CSV
                df.to_csv('../dataFiles/ICs.txt', header=None, index=False)
                print(f"Best model saved with loss: {best_loss}")

            elif anneal and torch.exp(-delta_loss / temperature) > np.random.random():
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

    print(f"closest_x = {closest_x}, closest_A = {closest_A}, closest_b = {closest_b}, closest_m = {closest_m}, closest_Omega = {closest_Omega}")
    print(f"x_start = {x_start}, A = {A}, b = {b}, m = {m}, Omega = {Omega}")
    print("best_loss =",best_loss)
    print("Current loss and time =", loss_func())
    #model = torch.jit.load("../NeuralNetworkData/xi_model_IC_2_point_225840410642715_.pt")
    #print("Current loss and time =", loss_func())
    if not automate:
        ans = input("Proceed? (y/n): ")
        if ans.lower() != 'y':
            exit()

    # Training loop
    global learning_rate
    learning_rate = 0.000125
    Algorithm = "adamax"
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
            brute_force(fine = False, coolingRate = 0.999, anneal = True, initial_temp = 1)
            if raiseBaseException:
                raise(KeyboardInterrupt)
        elif Algorithm == "newton":
            newton_method()
            if raiseBaseException:
                raise(KeyboardInterrupt)
        else:
            while criterion():
                if epoch >= 1000:
                    return best_loss
                optimizer.zero_grad()  # Zero the gradients
                loss_value, t_value = loss_func()
                
                # Check if the current loss is the best (lowest)
                if loss_value.item() < best_loss:
                    best_loss = loss_value.item()  #Update best loss
                    best_t_value = t_value #Update corresponding best time
                    torch.save(model.state_dict(), new_model_path if new_model_path.endswith(".pth") else new_model_path+"h")  # Save the best model
                    if saveLibTorch:
                        traced_script_module = torch.jit.trace(model, example)
                        traced_script_module.save(new_model_path.replace(".pth",".pt"))
                        print("libtorch version saved")
                    df = add_or_update_row(df, {'x_0': x_start, 'A': A, 'b': b, 'm': m, 'Omega': Omega, 'best_loss': best_loss, 'best_time': best_t_value})
                    if best_t_value != t_test_values[-1]:
                        print(f"New best t value = {best_t_value}")
                    try:
                        t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                    except:
                        t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                    # Write to CSV
                    df.to_csv('../dataFiles/ICs.txt', header=None, index=False)
                    print(f"Best model saved with loss: {best_loss}")

                loss_value.backward()  # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #        optimizer.step()  # Update weights
                if Algorithm == "lbfgs":
                    optimizer.step(closure)
                else:
                    optimizer.step()
                
                print(f'Epoch {epoch}, Loss: {loss_value.item()}')

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
            if raiseBaseException:
                raise(KeyboardInterrupt)

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
        if useLibTorch:
            model = torch.jit.load(new_model_path)
        else:
            model.load_state_dict(torch.load(new_model_path))#, weights_only=True))
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
        plt.title(f'$x_0$ = {x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
        plt.legend()
        plt.savefig("trajectory_data.svg")
        system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf trajectory_data.svg")
        system("rm trajectory_data.svg")
        system(f"open trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf")
        system(f"cp trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
        system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.png")
        system(f"open ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.png")
        plt.close()
        
        if produceInverse:
            x_values_inv = [-i for i in x_values]
            v_values_inv = [-i for i in v_values]
            a_values_inv = [-i for i in a_values]
            xi_values_inv = [-i for i in xi_values]
            plt.plot(t_test_values, x_values_inv, label='x(t) [m]', color='blue')
            plt.plot(t_test_values, v_values_inv, label='v(t) [m/s]', color='green', linestyle=':')
            plt.plot(t_test_values, a_values_inv, label='a(t) [m/$s^2$]', color='purple', linestyle='-.')
            plt.plot(t_test_values, xi_values_inv, label=r'$\xi(t)$', color='red', linestyle='--')
            #    plt.axhline(y=0.5, color='black', linestyle='--', alpha = 0.2)  # Red dashed line at y = 0.5
            #    plt.axhline(y=0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.01
            #    plt.axhline(y=-0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = -0.01
            plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.0
            
            plt.xlabel('t')
            plt.title(f'$x_0$ = {-x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
            plt.legend()
            plt.savefig("trajectory_data.svg")
            system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf trajectory_data.svg")
            system("rm trajectory_data.svg")
            system(f"open trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf")
            system(f"cp trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
            system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.png")
            system(f"open ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.png")

            plt.close()
        
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
    #    gold_dot, = ax.plot([], [], 'yo', markersize=8, label = "$x^*$")

        # Set plot limits and labels
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(0, 3)
        ax.set_xlabel("x")
        ax.set_ylabel("Potential")
        ax.legend()
        
        ax.set_title(f'$x_0$ = {x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
        sech = lambda x: 1/np.cosh(x)
        
        def potential(x, xi):
            V_MT = 0.5 * Omega * Omega * x * x
            temp = sech(A * (x - xi))
            V_SECH = A * A * temp * temp
            return V_MT + V_SECH

        def potential(x, xi):
            V_MT = 0.5 * Omega * Omega * x * x
            temp = sech(b * (x - xi))
            V_SECH = A * temp * temp
            return V_MT + V_SECH
        # Initialization function
        def init():
    #        gold_dot.set_data([], [])
            dot.set_data([], [])
            curve.set_data([], [])
            return dot, curve#, gold_dot

            # Update function
        def update(i):
            t = (i) * (best_t_value / (N - 1))  # Calculate current time
            y_values = potential(x_range, xi_values[i])
            dot.set_data([x_values[i]], [potential(x_values[i], xi_values[i])])
            curve.set_data(x_range, y_values)
    #        gold_dot.set_data([x_star], [potential(x_star, xi_values[i])])
            ax.set_title(f"$x_0$ = {x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
            return dot, curve#, gold_dot

        # Create animation
        fps = N/best_t_value

        ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval = 1000/fps)
        
        # Save the animation
        ani.save(f"../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4", writer=animation.FFMpegWriter(fps=2*fps))
        system(f"open ../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4")
        print(f"Movie saved as '../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4'")
        plt.close()

        if produceInverse:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Initialize plot elements
            dot, = ax.plot([], [], 'ro', markersize=8, label = "Particle")
            curve, = ax.plot([], [], 'b-', lw=2)
    #        gold_dot, = ax.plot([], [], 'yo', markersize=8, label = "$x^*$")

            # Set plot limits and labels
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(0, 3)
            ax.set_xlabel("x")
            ax.set_ylabel("Potential")
            ax.legend()
            ax.set_title(f'$x_0$ = {-x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
            sech = lambda x: 1/np.cosh(x)
            def update(i):
                t = (i) * (best_t_value / (N - 1))  # Calculate current time
                y_values = potential(x_range, -xi_values[i])
                dot.set_data([-x_values[i]], [potential(-x_values[i], -xi_values[i])])
                curve.set_data(x_range, y_values)
    #            gold_dot.set_data([x_star], [potential(-x_star, -xi_values[i])])
                ax.set_title(f"$x_0$ = {-x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
                return dot, curve#, gold_dot
            ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval = 1000/fps)
        
            # Save the animation
            ani.save(f"../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4", writer=animation.FFMpegWriter(fps=2*fps))
            system(f"open ../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4")
            print(f"Movie saved as '../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_.mp4'")

        plt.close()


master_func(m=1.3)
exit()
for A in np.arange(1.1, 2.1, 0.1):
    master_func(A=round(A,2))
##    start = time()
##    result = master_func(A=A)
##    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
##    with open("result_times.txt", "a") as f:
##        f.write(f"Time from A = {A-0.1:.2f} to {A:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
for m in np.arange(1.1, 2.1, 0.1):
    master_func(m=round(m,2))
#    start = time()
#    result = master_func(m=m)
#    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
#    with open("result_times.txt", "a") as f:
#        f.write(f"Time from m = {m-0.1:.2f} to {m:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
for b in np.arange(1., 2.1, 0.1):
    master_func(b=round(b,2))
#    start = time()
#    result = master_func(b=b)
#    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
#    with open("result_times.txt", "a") as f:
#        f.write(f"Time from b = {b-0.1:.2f} to {b:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
for Omega in np.arange(0.3, 1.3, 0.1):
    master_func(Omega=round(Omega,2))
#    start = time()
#    result = master_func(Omega=Omega)
#    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
#    with open("result_times.txt", "a") as f:
#        f.write(f"Time from Omega = {Omega-0.1:.2f} to {Omega:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
    



#../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/
#../movies/trajectory_trap_plus_sech_squared/
#/Users/edwardfinkelstein/RCPDE/EdwardF/scripts/result_times.txt
