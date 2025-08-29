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
from scipy.interpolate import UnivariateSpline
from warnings import filterwarnings
filterwarnings('ignore')

def master_func_learn_ivp_ode(m = 1.0, Omega = 0.2, A = 1.0, b = 1.0, load_model = True, base_model = "", simulate_only = {"simulate_only": False, "xStart": 0}, T = 10, v_start = 0.0, movie_x_lims = None, movie_y_lims = None, use_Variational_Potential = False, automate = True, produceInverse = False, saveLibTorch = True, useLibTorch = True, epsilon = 0.0, lrScheduler = False, learning_rate = 1e-5, weight_decay = 1e-4, Algorithm = "adam", fine = False, coolingRate = 0.999, anneal = True, initial_temp = 100, amsgrad = False):
    #Setting the random seeds!!!
    np.random.seed(42)
    torch.manual_seed(42)
    ICsFile = '../dataFiles/ICsVariational.txt' if use_Variational_Potential else '../dataFiles/ICs.txt'
    A_times_b = A*b
    Omega_squared = Omega*Omega
    b_squared = b*b
    def sech(x):
        if isinstance(x, torch.Tensor):
            return 1 / torch.cosh(x)
        elif isinstance(x, (float, np.float64, int, np.ndarray)):
            return 1 / np.cosh(x)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
    
    def tanh(x):
        if isinstance(x, torch.Tensor):
            return torch.tanh(x)
        elif isinstance(x, (float, np.float64, int, np.ndarray)):
            return np.tanh(x)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

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
            (temp_df['Omega'] == new_row_data['Omega'])]
            
        if matching_rows.empty:
            # Add a new row
            df.loc[len(df)] = new_row_data
        else:
            # Overwrite existing row
            row_index = matching_rows.index[0]  # Get the index of the first (and only) match
            df.loc[row_index] = new_row_data

        return df

    T = T       # Final time
    dt = 0.01      # Time step
    x_star = 0.0   # Final position sought
    to_time = {"timed": False, "time": 3600}
    to_loss = {"loss thresholded": True, "threshold": 1.4e-2}
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
    
    def is_number(x):
        if isinstance(x, (int, float, np.number)):
            return True
        elif isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 0:
            return True
        return False

    def potential(x, xi):
        V_MT = 0.5 * Omega * Omega * x * x
        temp = torch.sech(b * (x - xi))
        V_SECH = A * temp * temp
        return V_MT + V_SECH

    # Function to compute the force (negative derivative of potential)
    def force(x, xi):
        temp = x - xi
        temp_sech = torch.sech(b * (temp))
        return -(Omega*Omega * x) + (2 * A*b * temp_sech*temp_sech * tanh(b * (temp)))
    
    def isclose_zero(x, atol=1e-12):
        if isinstance(x, np.ndarray):
            return np.isclose(x, 0.0, atol=atol)
        elif 'torch' in str(type(x)):
            import torch
            return torch.isclose(x, torch.tensor(0.0, dtype=x.dtype, device=x.device), atol=atol)
        else:
            return abs(x) < atol

    # Define the effective potential function
    def Veff_plus_trap(x, xi=0):
        """
        Compute the variational effective potential U_eff + trap term.

        Parameters:
        -----------
        x : float or array-like
            Position(s) to evaluate the potential at.
        xi : float
            Control shift value (center of the hill), default is 0.
        
        Returns:
        --------
        Potential : float or array-like
            Evaluated potential energy at each position.
        """
        X = x - xi
        X_squared = X*X
        U_taylor = (
            - (64 * A_times_b*b_squared / 105) * X_squared
            + (16 * A_times_b / 15)
            + (2 * Omega_squared * b / 3) * X_squared
        )
        module = torch if isinstance(x, torch.Tensor) else np
        exp_term = module.exp(2 * b * X) - 1

        # Avoid divide-by-zero instability
        exp_term = module.where(isclose_zero(exp_term), 1e-12, exp_term)
        exp_term_squared = exp_term*exp_term
        exp_term_fourth = exp_term_squared*exp_term_squared

        # Compute each term
        term1 = (-256 * A*b_squared * X) / (exp_term_fourth*exp_term)
        term2 = (-256 * A_times_b * (2 * b * X - 1)) / (exp_term_squared*exp_term)
        term3 = (-128 * A_times_b * (5 * b * X - 1)) / (exp_term_fourth)
        term4 = (-32  * A_times_b * (12 * b * X - 13)) / (3 * exp_term_squared)
        term5 =  (32  * A_times_b) / (3 * exp_term)
        trap  =  (2 * Omega_squared * b * X_squared) / 3

        U_eff = term1 + term2 + term3 + term4 + term5 + trap
        # Apply patch
        use_taylor = (module.abs(X) < epsilon)
        return np.where(use_taylor, U_taylor, U_eff)

    # Define the effective force function
    def Force_eff_plus_trap(x, xi=0):
        """
        Compute the variational force (effective + trap) for scalar or array input.
        
        Parameters:
        -----------
        x : float or array-like
            Position(s) to evaluate the force at.
        xi : float
            Control shift value (center of the hill), default is 0.
        
        Returns:
        --------
        Force : float or array-like
            Evaluated force at each position.
        """
        X = x - xi
        #F_{\text{Taylor}}(\mathcal{X}) = & \dfrac{128 A \mathcal{A}_0^{3} \mathcal{X}}{105} - \dfrac{4 \Omega^{2} \mathcal{A}_0 \mathcal{X}}{3}
        F_taylor = (
              ((128 * A_times_b*b_squared * X) / 105)
            - ((4 * Omega_squared * b * X) / 3)
        )
        module = torch if isinstance(X, torch.Tensor) else np
        exp_term = module.exp(2 * b * X) - 1

        # Avoid divide-by-zero and warnings
        exp_term = module.where(isclose_zero(exp_term), 1e-12, exp_term)

        # Compute each term
        term1 = -2560 * A * b**3 * X / exp_term**6
        term2 = -1280 * A * b**2 * (6 * b * X - 1) / exp_term**5
        term3 = -64   * A * b**2 * (8 * b * X - 11) / exp_term**2
        term4 = -128  * A * b * (64 * b * X - 25) / exp_term**4
        term5 = -128  * A * b**2 * (84 * b * X - 61) / (3 * exp_term**3)
        term6 =  64   * A * b**2 / (3 * exp_term)
        trap  = -4/3 * b * Omega*Omega * x

        Feff = term1 + term2 + term3 + term4 + term5 + term6 + trap
        # Apply patch
        use_taylor = (module.abs(X) < epsilon)
        return module.where(use_taylor, F_taylor, Feff)
        
    potential = Veff_plus_trap if use_Variational_Potential else potential
    force = Force_eff_plus_trap if use_Variational_Potential else force
    min_force = lambda x: force(x, 0)

    smoothness_penalty_factor = 1e-3 #α: penalty for lack of smoothness of xi
    time_penalty_factor = 1e-3 #β: penalty for taking longer
    velocity_penalty = 1e-3 #γ: penalty for max(abs(v))
    xi_penalty = 1e-3 #δ: penalty for max(abs(xi))
    x_star_x_diff_mse_penalty = 1 #ε: penalty for (x_star_x_diff*x_star_x_diff) term in MSE in loss_func
    v_mse_penalty = 1 #ζ: penalty for (v*v) term in MSE in loss_func
    x_star_xi_diff_mse_penalty = 1 #η: penalty for (x_star_xi_diff*x_star_xi_diff) term in MSE in loss_func
    
    t_values = np.linspace(1e-8, T, int(T / dt))
    delta_t = t_values[1] - t_values[0]
    global t_test_values
    t_test_values = np.linspace(1e-8, T, int(T / dt))

    # Initial guess for the root (you might need to adjust this)
    x0 = 2.0

    # Find the root
    root, info, ier, mesg = fsolve(min_force, x0, full_output=True)

    if ier == 1:
        residual = min_force(root[0])
        tol = 1e-5
        assert np.isclose(residual, 0.0, atol=tol), f"fsolve claimed convergence but residual {residual} exceeds tolerance {tol}"
        print(f"Root found: {root[0]}")
        x_start, v_start = float(root[0]), v_start
    else:
        print(f"Root finding failed: {mesg}")
        exit()

    x_start = round(float(x_start), 6)

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
    global best_loss, best_t_value, df
    best_loss = np.inf
    best_t_value = T
    df = None
    try:
        df = pd.read_csv(ICsFile, names=("x_0", "A", "b", "m", "Omega", "best_loss", "best_time"))
    except FileNotFoundError:
        # Create the file if it doesn't exist
        with open(ICsFile, 'w') as f:
            print("File created successfully.")
        df = pd.DataFrame(columns=("x_0", "A", "b", "m", "Omega", "best_loss", "best_time"))

    parameters = df[(df['x_0']==x_start) & (df['A']==A) & (df['b']==b) & (df['m']==m) & (df['Omega']==Omega)]
    closest_x, closest_A, closest_b, closest_m, closest_Omega = x_start, A, b, m, Omega

    if len(parameters):
        print(f"parameters = {parameters}")
        best_loss, best_t_value = parameters['best_loss'].item(), parameters['best_time'].item()
        try:
            t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
        except:
            t_test_values = None if simulate_only["simulate_only"] else t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
    elif len(df):
        #Extract the row with the closest 'x_0', 'A', 'b', 'm', 'Omega' to (x_start, A, b, m, Omega) based on euclidean distance
        # Create a new DataFrame with the relevant columns for distance calculation
        relevant_cols = ['x_0', 'A', 'b', 'm', 'Omega']
        temp_df = df[relevant_cols]

        # Calculate Euclidean distance between each row and the target values
        temp_df['distance'] = np.linalg.norm(temp_df - [x_start, A, b, m, Omega], axis=1)
        # Find the index of the row with the minimum distance
        closest_row_idx = temp_df['distance'].idxmin()
        parameters = temp_df.iloc[closest_row_idx]
        closest_x, closest_A, closest_b, closest_m, closest_Omega = round(parameters['x_0'], 6), round(parameters['A'], 6), round(parameters['b'], 6), round(parameters['m'], 6), round(parameters['Omega'], 6)

    # Load or instantiate the model
    file_suffix = ("_Variational_" if use_Variational_Potential else "_")
    model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(closest_x)}_{flt_to_str(round(closest_A, 6))}_{flt_to_str(round(closest_b, 6))}_{flt_to_str(round(closest_m, 6))}_{flt_to_str(round(closest_Omega, 6))}{file_suffix}.pth" if not base_model else base_model
    new_model_path = f"../NeuralNetworkData/xi_model_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pth"
    print(f"model_path = {model_path}")
    print(f"new_model_path = {new_model_path}")
    if closest_x == x_start and round(closest_A, 4) == round(A, 4) and round(closest_b, 4) == round(b, 4) and round(closest_m, 4) == round(m, 4) and round(closest_Omega, 4) == round(Omega, 4):
        assert(model_path == new_model_path or base_model)
    if useLibTorch:
        new_model_path = new_model_path.replace(".pth", ".pt")
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
        if useLibTorch:
            new_model_path = new_model_path.replace(".pth", ".pt")
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

    def rk4_step_with_accel(x, v, xi_t, dt):
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

        # RK4 "averaged" acceleration over this step
        a_avg = (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0

        return x_new, v_new, a_avg
    
    # Define the xi function using the neural network
    def xi(t):
        t_input = torch.tensor([[t]], dtype=torch.float32)  # Convert to tensor
        return model(t_input)[0, 0]  # Get the output from the model

    def simulate_trajectory(xStart = None):
        xStart = x_start if not xStart else xStart
        x, v = (x_start if not xStart else xStart), v_start
        x_values = [x]
        print(f"x_values[-1] = {x_values[-1]}, v = {v}")
        for i, t in enumerate(t_values[1:]):
            x, v = rk4_step(x, v, torch.tensor([[0]], dtype=torch.float32)[0, 0], dt)
            x_values.append(x.detach().numpy())  # Use detach()

        return x_values, t_values

    if simulate_only["simulate_only"]:
        return simulate_trajectory(simulate_only["xStart"])

    def loss_func():
        x, v = x_start, v_start
        smoothness_penalty = 0.0  # Initialize smoothness penalty
        xi_values_temp = []  # Temporary storage for xi values to calculate smoothness
        v_values_temp = [] # Temporary storage for v values to calculate max velocity
        best_loss_ = np.inf
        best_time = np.inf
        best_time_idx = np.inf
        
        for i, t in enumerate(t_values):
            xi_t = xi(t) if i > 0 else torch.tensor([[0]], dtype=torch.float32)[0, 0]  # Compute xi(t) at time t, assuming it is 0 at t ~ 0
            xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
            x, v = rk4_step(x, v, xi_t, dt)
            v_values_temp.append(abs(v))
            
    #        print(f"i = {i}, x = {x}, v = {v}")
    #        print(f"t = {t}, xi(t) = {xi_values_temp[-1]}\n")
                        
            x_star_x_diff = (x_star - x)
            x_star_xi_diff = (x_star - xi(t))
#            xi_0 = xi(0)
            
            if i > 1:
                MSE = x_star_x_diff_mse_penalty * (x_star_x_diff*x_star_x_diff) + v_mse_penalty * (v*v) + x_star_xi_diff_mse_penalty * (x_star_xi_diff*x_star_xi_diff)# + (xi_0*xi_0)
                if MSE < best_loss_:
    #                print(f"x_star_x_diff*x_star_x_diff = {x_star_x_diff*x_star_x_diff}")
    #                print(f"v*v = {v*v}")
    #                print(f"x_star_xi_diff*x_star_xi_diff = {x_star_xi_diff*x_star_xi_diff}")
    #                print(f"xi_0*xi_0 = {xi_0*xi_0}")
                    best_time = t
                    best_loss_ = MSE
                    best_time_idx = i
        
        if best_time_idx == np.inf:
            best_time_idx = 2
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

    # f(x)|_{expanded about x=a} = f(a) + (x-a)f'(a) + ((x-a)^2)/2)*f''(a) + ...
    # Define Newton's method function
    def newton_method():
        global best_loss, df, t_test_values, best_t_value
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
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A, 6), 'b': round(b, 6), 'm': round(m, 6), 'Omega': round(Omega, 6), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    print(f"New best t value = {best_t_value}")
                try:
                    t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                except:
                    t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                # Write to CSV
                df.to_csv(ICsFile, header=None, index=False)
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
        global best_loss, df, t_test_values, best_t_value
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
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A, 6), 'b': round(b, 6), 'm': round(m, 6), 'Omega': round(Omega, 6), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    print(f"New best t value = {best_t_value}")
                try:
                    t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                except:
                    t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                # Write to CSV
                df.to_csv(ICsFile, header=None, index=False)
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
    if not automate:
        ans = input("Proceed? (y/n): ")
        if ans.lower() != 'y':
            exit()
    
    optimizer = None
    if Algorithm == "lbfgs":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, max_iter=20, history_size=50)
    elif Algorithm == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad = amsgrad)
    elif Algorithm == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif Algorithm == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate)
    elif Algorithm == "adafactor":
        optimizer = optim.Adafactor(model.parameters(), lr=learning_rate)
    elif Algorithm == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay = weight_decay, amsgrad = amsgrad)
    elif Algorithm == "asgd":
        optimizer = optim.ASGD(model.parameters(), lr=learning_rate)
    elif Algorithm == "sparse adam":
        optimizer = optim.SparseAdam(model.parameters(), lr=learning_rate)
    
    scheduler = None
    if lrScheduler and optimizer:
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate/10, max_lr=learning_rate, step_size_up=50)
    plot_progress = False
    epoch = 0
    start_time = time()

    try:
        if Algorithm == "brute force":
            # Define constants and call the simulated annealing function
            brute_force(fine = fine, coolingRate = coolingRate, anneal = anneal, initial_temp = initial_temp)
            if raiseBaseException:
                raise(KeyboardInterrupt)
        elif Algorithm == "newton":
            newton_method()
            if raiseBaseException:
                raise(KeyboardInterrupt)
        else:
            while criterion():
#                if epoch >= 1000:
#                    return {"result": f'target not achieved, best loss after epoch 1000 = {best_loss}', "closest_x": closest_x, "closest_A": closest_A, "closest_b": closest_b, "closest_m": closest_m, "closest_Omega": closest_Omega}
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
                    df.to_csv(ICsFile, header=None, index=False)
                    print(f"Best model saved with loss: {best_loss}")

                loss_value.backward()  # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #        optimizer.step()  # Update weights
                if Algorithm == "lbfgs":
                    optimizer.step(closure)
                else:
                    optimizer.step()
                if lrScheduler:
                    scheduler.step()
                
                print(f'Epoch {epoch}, Loss: {loss_value.item()}')

                if plot_progress and epoch % 5 == 0:
                    x_values, v_values, xi_values, a_values = [], [], [], []
                    x, v = x_start, v_start  # Initial conditions
                    for i, t in enumerate(t_test_values):
                        xi_t = xi(t) if i > 0 else torch.tensor([[0]], dtype=torch.float32)[0, 0]
                        xi_values.append(xi_t.detach().numpy())
                        a = force(x, xi_t) / m
                        x_values.append(x.detach().numpy())  # Use detach()
                        v_values.append(v.detach().numpy())  # Use detach()
                        a_values.append(a.detach().numpy())
                        x, v = rk4_step(x, v, xi_t, dt)
                        
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
        if useLibTorch and os.path.exists(new_model_path):
            model = torch.jit.load(new_model_path)
        else:
            model.load_state_dict(torch.load(new_model_path))#, weights_only=True))
        print("New model path loaded =",new_model_path)
        xi = lambda t: model(torch.tensor([[t]], dtype=torch.float32))[0, 0] # Get the output from the model
        # Save data to CSV
        data_path = "../dataFiles/trajectory_data.csv"
        x_values, v_values, a_values, xi_values = [], [], [], []
        x, v = x_start, v_start # Initial conditions
        a = force(torch.tensor(x_start), xi(0.0)) / m
        
        print(f"t_values.shape = {t_values.shape}")
        if ((isinstance(t_test_values, np.ndarray) and not len(t_test_values)) or (not isinstance(t_test_values, np.ndarray) and not t_test_values)):
            assert best_t_value, f"best_t_value = {best_t_value}"
            try:
                t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
            except:
                t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]

        for i, t in enumerate(t_test_values):
            xi_t = xi(t) if i > 0 else torch.tensor([[0]], dtype=torch.float32)[0, 0]
            xi_values.append(xi_t.detach().numpy())
            x, v = rk4_step(x, v, xi_t, dt)
            a = force(x, xi_t) / m
#            print(f"x = {x}, xi = {xi_t}, a = {a}")
            x_values.append(x.detach().numpy())
            v_values.append(v.detach().numpy())
#            a_values.append(a.detach().numpy())
#            assert(len(a_values) == len(x_values))
        print(f"max(v_values) = {max(v_values)}")
        spline = UnivariateSpline(t_test_values, v_values)
        a_values = spline.derivative()(t_test_values)
        assert(len(a_values) == len(x_values))

        with open(data_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
            for i in range(len(t_test_values)):
                writer.writerow([t_values[i], xi_values[i].item(), x_values[i].item(), v_values[i].item()])
        print("Data saved to CSV.")
        title_sep = '\n' if use_Variational_Potential else ' '

#        print("x_values[-10:], v_values[-10:], a_values[-10:] = ", x_values[-10:], v_values[-10:], a_values[-10:], sep='\n')
        plt.plot(t_test_values, x_values, label='x(t) [m]', color='blue')
        plt.plot(t_test_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
        plt.plot(t_test_values, a_values, label='a(t) [m/$s^2$]', color='purple', linestyle='-.')
        plt.plot(t_test_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
    #    plt.axhline(y=0.5, color='black', linestyle='--', alpha = 0.2)  # Red dashed line at y = 0.5
    #    plt.axhline(y=0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.01
    #    plt.axhline(y=-0.01, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = -0.01
        plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.2, linewidth=0.5)  # Thin black dashed line at y = 0.0

        plt.xlabel('t')
        b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
        plt.title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}")
        plt.legend()
        plt.savefig("trajectory_data.svg")
        system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf trajectory_data.svg")
        system(f"open trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf")
        system(f"cp trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
        system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.png")
        system(f"open ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.png")
        print(f"image ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.png saved")
        system(f"rm trajectory_data.svg trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf ")
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
            b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
            plt.title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {-x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}")
            plt.legend()
            plt.savefig("trajectory_data.svg")
            system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf trajectory_data.svg")
            system(f"rm trajectory_data.svg trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf")
            system(f"open trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf")
            system(f"cp trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
            system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.png")
            system(f"open ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.png")

            plt.close()
        
        if not automate:
            answer = input("Movie (y/n)? ")
            if not answer.lower().startswith('y'):
                print("Movie creation skipped.")
                exit()
            
        N = len(t_test_values)
        x_range = np.linspace(-10, 10, 1000)
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
        b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
        ax.set_title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {-x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
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
            b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
            ax.set_title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
            if movie_x_lims:
                ax.set_xlim(movie_x_lims)
            if movie_y_lims:
                ax.set_ylim(movie_y_lims)
            return dot, curve#, gold_dot

        # Create animation
        fps = N/best_t_value
#        print(xi_values, type(xi_values))
        print(f"Omega = {Omega}, b = {b}, A = {A}, xi_0 = {xi_values[0]}")

        ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval = 1000/fps)
        
        # Save the animation
        ani.save(f"../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4", writer=animation.FFMpegWriter(fps=2*fps))
        system(f"open ../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4")
        print(f"Movie saved as '../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4'")
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
            b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
            ax.set_title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {-x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
            def update(i):
                t = (i) * (best_t_value / (N - 1))  # Calculate current time
                y_values = potential(x_range, -xi_values[i])
                dot.set_data([-x_values[i]], [potential(-x_values[i], -xi_values[i])])
                curve.set_data(x_range, y_values)
    #            gold_dot.set_data([x_star], [potential(-x_star, -xi_values[i])])
                b_symb = r'$\mathcal{A}_0$' if use_Variational_Potential else '$b$'
                ax.set_title((r'$U_{\mathrm{patched}}$: ' if use_Variational_Potential else '') + f"$x_0$ = {-x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f},{title_sep}{b_symb} = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
                if movie_x_lims:
                    ax.set_xlim(movie_x_lims)
                if movie_y_lims:
                    ax.set_ylim(movie_y_lims)
                return dot, curve#, gold_dot
            ani = animation.FuncAnimation(fig, update, frames=N, init_func=init, blit=True, interval = 1000/fps)
        
            # Save the animation
            ani.save(f"../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4", writer=animation.FFMpegWriter(fps=2*fps))
            system(f"open ../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4")
            print(f"Movie saved as '../movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 6))}_{flt_to_str(round(b, 6))}_{flt_to_str(round(m, 6))}_{flt_to_str(round(Omega, 6))}{file_suffix}.mp4'")

        plt.close()
        return {"result": "target achieved", "closest_x": closest_x, "closest_A": closest_A, "closest_b": closest_b, "closest_m": closest_m, "closest_Omega": closest_Omega, "learning_rate": learning_rate}

if __name__=='__main__':
#    On Saturday, July 12, 2025, 9:53 pm EST, cat -n result_times.txt yields 120 lines
#    master_func_learn_ivp_ode(m = 1.0, A = 1.0, b = 1.0, Omega = 0.2, load_model = True, base_model = "xi_model_IC_2_point_438068_1_point_08_0_point_8_1_point_0_0_point_23_.pth", simulate_only = {"simulate_only": False, "xStart": 0}, T = 10, v_start = 0.0, movie_x_lims = None, movie_y_lims = None, use_Variational_Potential = True, automate = True, produceInverse = False, saveLibTorch = True, useLibTorch = True)

    start_b, end_b = 1.0, 0.75
    start_Ω, end_Ω = 0.2, 0.18
    b_space = np.linspace(start_b, end_b, 25)[2:]
    Ω_space = np.linspace(start_Ω, end_Ω, 25)[2:]
    db = b_space[1] - b_space[0]
    dΩ = Ω_space[1] - Ω_space[0]
        
    for b, Ω in zip(b_space, Ω_space):
        start = time()
        result = master_func_learn_ivp_ode(m = 1.0, A = 1.0, b = b, Omega = Ω, load_model = True, base_model = "", simulate_only = {"simulate_only": False, "xStart": 0}, T = 10, v_start = 0.0, movie_x_lims = None, movie_y_lims = None, use_Variational_Potential = True, automate = True, produceInverse = False, saveLibTorch = True, useLibTorch = True, epsilon = 0.25/b, lrScheduler = False, learning_rate = 1e-4, weight_decay = 1e-4, Algorithm = "adam", fine = False, coolingRate = 0.999, anneal = False, initial_temp = 1, amsgrad = True)
        achieved = result["result"]
        learning_rate = result["learning_rate"]
        with open("result_times_variational.txt", "a") as f:
            prev_b = b-db if b != start_b else result['closest_b']
            prev_Ω = Ω-dΩ if Ω != start_Ω else result['closest_Omega']
            f.write(f"Time from b = {prev_b:.4f} to {b:.4f}, Ω = {prev_Ω:.4f} to {Ω:.4f}, = {time() - start:.4f} with learning rate {learning_rate}, {achieved}\n")
            


#    start_A, end_A = 1.1, 1.08
#    start_b, end_b = 1.0, 0.8
#    start_Ω, end_Ω = 0.2, 0.23
#    A_space = np.linspace(start_A, end_A, 21)
#    b_space = np.linspace(start_b, end_b, 21)
#    Ω_space = np.linspace(start_Ω, end_Ω, 21)
#    dA = A_space[1] - A_space[0]
#    db = b_space[1] - b_space[0]
#    dΩ = Ω_space[1] - Ω_space[0]
#        
#    for A, b, Ω in zip(A_space, b_space, Ω_space):
#        start = time()
#        result = master_func_learn_ivp_ode(m = 1.0, A=round(A, 6), b = round(b, 6), Omega = round(Ω, 6), load_model = True, base_model = "", simulate_only = {"simulate_only": False, "xStart": 0}, T = 10, v_start = 0.0, movie_x_lims = None, movie_y_lims = None, use_Variational_Potential = False, automate = True, produceInverse = False, saveLibTorch = True, useLibTorch = True)
#        achieved = result["result"]
#        with open("result_times.txt", "a") as f:
#            prev_A = A-dA if A != start_A else result['closest_A']
#            prev_b = b-db if b != start_b else result['closest_b']
#            prev_Ω = Ω-dΩ if Ω != start_Ω else result['closest_Omega']
#            f.write(f"Time from A = {prev_A:.4f} to {A:.4f}, b = {prev_b:.4f} to {b:.4f}, Ω = {prev_Ω:.4f} to {Ω:.4f}, = {time() - start:.4f} with learning rate {learning_rate}, {achieved}\n")
    
#../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/
#../movies/trajectory_trap_plus_sech_squared/
#/Users/edwardfinkelstein/RCPDE/EdwardF/scripts/result_times.txt

#ls -t -r ../movies/trajectory_trap_plus_sech_squared/*mp4 | tail -n 10| xargs
#ls -t -r ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/*png | tail -n 10 | xargs
