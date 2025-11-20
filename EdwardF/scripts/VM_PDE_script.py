from google.colab import drive
drive.mount('/content/drive')
DRIVE_BASE_PATH = '/content/drive/MyDrive/BrightSolitonPDE'

#IDForModelWeights = 1YREdAZabLwQZyQx638yIaZpJPmscuClk
#IDForTxtFile = 1thnAE0XN_Lhht2ryKlsU5D1Y9KI2qIN0


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
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
from torch import diag
from warnings import filterwarnings
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from glob import glob
from scipy.optimize import minimize
filterwarnings('ignore')

def master_func_learn_ivp_pde(m = 1.0, Omega = 0.2, A = 1.0, b = 1.0, load_model = True, base_model = "", no_print = False, get_model_loss_value = False, optimize_A_b_Omega_m = False, optimize_A_b_Omega_m_iterations = np.inf, simulate_only = {"simulate_only": False, "xStart": 0, "store mass values": False}, T = 100, v_start = 0.0, interpolate = True, add_kick = False):
    #Setting the random seeds!!!
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    def sech(x):
        if isinstance(x, torch.Tensor):
            return 1 / torch.cosh(x)
        elif isinstance(x, (float, np.float64, int, np.ndarray)):
            return 1 / np.cosh(x)
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
    sigma = 1.0     # Width of the (Gaussian) potential, not used currently
    T = T        # Final time
    dt = 0.001      # Time step
    num_steps = int(T / dt)

    x_star = 0.0   # Final position sought
    v_th = 0.01    # Velocity threshold, not used currently
    x_th = 0.01    # Position threshold, not used currently
    to_time = {"timed": False, "time": 3600}
    to_loss = {"loss thresholded": True, "threshold": 1.1e-4}
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
        nonlocal A, b, Omega
        V_MT = 0.5 * Omega * Omega * x * x
        temp = torch.sech(b * (x - xi))
        V_SECH = A * temp * temp
        return V_MT + V_SECH

    def sech_potential(x, xi):
        temp = torch.sech(b * (x - xi))
        return A*temp*temp
#        return torch.zeros_like(x)

    # Function to compute the force (negative derivative of potential)
    def force(x, xi):
        return -(Omega**2 * x) + (2 * A**3 * torch.sech(A * (x - xi))**2 * torch.tanh(A * (x - xi)))

    def force(x, xi):
        return -(Omega**2 * x) + (2 * A*b * torch.sech(b * (x - xi))**2 * torch.tanh(b * (x - xi)))

    def min_force(x):
        nonlocal A, b, Omega
        return (Omega**2 * x) - (2 * A*b * np.cosh(b * (x))**(-2) * np.tanh(b * (x)))

    #criterion = lambda: True if not to_time["timed"] else time() - start_time < to_time["time"]
    automate = True
    produceInverse = False
    saveLibTorch = True
    useLibTorch = True

    smoothness_penalty_factor = 1e-5 # penalty for lack of smoothness of xi
    time_penalty_factor = 1e-5 #penalty for taking longer
    velocity_penalty = 1e-5 #penalty for max(abs(v))
    xi_penalty = 1e-5 #penalty for max(abs(xi))
    x_star_x_diff_mse_penalty = 1 #penalty for (x_star_x_diff*x_star_x_diff) term in MSE in loss_func
    v_mse_penalty = 1 #penalty for (v*v) term in MSE in loss_func
    x_star_xi_diff_mse_penalty = 1 #penalty for (x_star_xi_diff*x_star_xi_diff) term in MSE in loss_func

    t_values = np.linspace(1e-8, T, num_steps)
    delta_t = t_values[1] - t_values[0]
    global t_test_values
    t_test_values = np.linspace(1e-8, T, num_steps)

    def get_x_start_and_v_start():
        nonlocal v_start
        # Initial guess for the root (you might need to adjust this)
        x0 = 1.0

        # Find the root
        root, info, ier, mesg = fsolve(min_force, x0, full_output=True)

        if ier == 1:
            root_min, potential_min = root[0], potential(root[0], 0)
            for root_val in root:
                potential_val = potential(root_val, 0)
                if potential_val < potential_min:
                    potential_min = potential_val
                    root_min = root_val
            if not no_print:
                print(f"Root found: {root_min}")
            x_start, v_start = float(root_min), v_start
            if not no_print:
                print(f"v_start = {v_start}")
        else:
            raise Exception(f"Root finding failed: {mesg}")
            exit()

        x_start = round(float(x_start), 4)
        return x_start, v_start

    x_start, v_start = get_x_start_and_v_start()
    min_x_start = 1
    if np.abs(x_start) < min_x_start:
        raise Exception(f"Error, must modify A, b, Omega so that |x_start| >= {min_x_start}")

    ###BEGIN NEWTON
    def refine_with_newton_helper(plot_steady_state = False, return_all = True, xStart = None, lr=1, scipySolve = True):
        nonlocal A, b, Omega, x_start, v_start
        xStart = x_start if not xStart else xStart.numpy().item()
        tol = 1e-10
        g = -1;                         # g = 1 is defocusing and g =-1 is focusing
        N = 401;                        # number of mesh points
        L = -10; R = 10;                # Left and right bounds of interval
        x = np.linspace(L, R, N)[:-1];    # Adjust for periodic boundary conditions
        N -= 1
        dx = x[1]-x[0];                 # mesh size
        point_5_over_dx_squared = (0.5 / dx**2)
        one_over_six = 1.0/6.0
#        assert(dt<0.7071067811865476*dx*dx)

        A_sol = 1; c = 0                     # Amplitude, vel. & position
        u0 = A_sol*sech(A_sol*(x - xStart))*np.exp(1j*c*x);    # initial condition (IC)
        V = potential(x, 0)

        U = np.concatenate((np.real(u0), np.imag(u0)))
        w_sol = 0.5#A_sol*A_sol*0.5 #temporal freq
    #    A_sol = 2; c = 0;
    #    w_sol = A*A*0.5 #temporal freq
    #    u0 = w_sol*sech(w_sol*(x))*torch.exp(1j*c*x);    # initial condition (IC)
    #    V = potential(x, 0)
    #    U = torch.cat([torch.real(u0), torch.imag(u0)], dim=0)

        err = np.inf

        # Define the discrete Laplacian with periodic boundary conditions
        ONE = np.ones(N)
        D2 = diags([ONE, -2 * ONE, ONE], [-1, 0, 1], shape=(N, N)).toarray()
        D2[0, -1] = D2[-1, 0] = 1  # Periodic boundary conditions
        D2 /= dx**2

        # Index for real and imaginary parts
        indR = slice(0, N)
        indI = slice(N, 2 * N)

        u = U[indR]+1j*U[indI];      # wrapping into a complex vector
        idx=np.where(np.isclose(abs(u), max(abs(u))))
        u_before = u.copy()

        def fresid(U):
            # Split real and imaginary parts
            Ur = U[indR]
            Ui = U[indI]
            # Compute the modulus squared of u
            U2 = Ur**2 + Ui**2
            common_term = (g * U2 + V + w_sol)
            return np.concatenate((-0.5 * (D2 @ Ur) + common_term * Ur,\
                                   -0.5 * (D2 @ Ui) + common_term * Ui))
        if scipySolve:
            U, infodict, ier, msg = fsolve(fresid, U, full_output=True)
            if not no_print:
                if ier == 1:
                    print("Solution found, error estimate:", np.linalg.norm(infodict['fvec']))
                else:
                    print("Solution may not have converged:", msg)
        else:
            # Main loop: checking Newton tolerance
            num_iter = 0
            while err > tol:
                # Split real and imaginary parts
                Ur = U[indR]
                Ui = U[indI]

                # Compute modulus squared of u
                U2 = Ur**2 + Ui**2

                # Right-hand side (RHS)
                common_term = (g * U2 + V + w_sol)
                Fr = -0.5 * (D2 @ Ur) + common_term * Ur
                Fi = -0.5 * (D2 @ Ui) + common_term * Ui
                F = fresid(U2, Ur, Ui)

                # Jacobian components
                J11 = -0.5 * D2 + np.diag(g * (3 * Ur**2 + Ui**2) + V + w_sol)
                J22 = -0.5 * D2 + np.diag(g * (Ur**2 + 3 * Ui**2) + V + w_sol)
                J12 = np.diag(2 * g * Ur * Ui)
                J = np.block([[J11, J12], [J12, J22]])

                # Newton correction
                DU = splu(J).solve(-F)  # Solve J * DU = -F
                U1 = U + lr*DU  # Update solution

                # Update error and solution
                err = np.linalg.norm(F)
                if not no_print:
                    print(f"err = {err}")
        #        if not no_print:
        #            print(f"Condition number of J: {torch.linalg.cond(J)}")
                U = U1
                num_iter += 1
                if num_iter > 100:
                    if not no_print:
                        print(f"Failed")
                    return [False]

        u = U[indR]+1j*U[indI];      # wrapping into a complex vector
        Maxu=max(abs(u));
        idx=np.where(np.isclose(abs(u), Maxu))


        if interpolate:
            # Compute the density (modulus squared of u)
            # density = u * np.conj(u)  # Equivalent to |u|^2

            # Calculate xmax (center of mass)
            #numerator = np.sum(x * density.real) * dx
            #denominator = np.sum(density.real) * dx

            # Compute the density (modulus squared of u)
            density = u * np.conj(u)  # Equivalent to |u|^2

            # Calculate xmax (center of mass)
            numerator = np.trapz(x * density.real, x)
            denominator = np.trapz(density.real, x)

            xmax = numerator / denominator
            if not no_print:
                print(f"xmax = {xmax}, xStart = {xStart}, xmax - xStart = {xmax - xStart}")
                print(f"x.shape = {x.shape}, u.shape = {u.shape}")
            # Create a spline interpolator
            interpolator = interp1d(x, u, kind='cubic', bounds_error=False, fill_value=np.nan)

            # Perform the displacement
            u_displaced = interpolator(x + (xmax - xStart))
    #        u_displaced = np.roll(u, 1)
    #        u_displaced *= A_sol/max(u_displaced)
    #        u_displaced = interpolator(x)

            # Replace NaN values with 0
            u = np.nan_to_num(u_displaced)

        if plot_steady_state:
            # Plot real part
    #        plt.plot(x.numpy(), u_before.real.numpy() + V.numpy(), label="$u_{\mathrm{real}}$ before Newton + $V$", color='purple')
            plt.plot(x, u_before.real, label="$u_{\mathrm{real}}(x,0)$ before Newton", color='purple')
            # Plot imaginary part
            plt.plot(x, u_before.imag, label="$u_{\mathrm{imag}}(x,0)$ before Newton", color='orange', linestyle='--')
            # Plot real part
    #        plt.plot(x.numpy(), Ur.numpy() + V.numpy(), label="$u_{\mathrm{real}}$ after Newton + $V$", color='blue')
            plt.plot(x, u.real, label="$u_{\mathrm{real}}(x,0)$ after Newton", color='blue')
            # Plot imaginary part
            plt.plot(x, u.imag, label="$u_{\mathrm{imag}}(x,0)$ after Newton", color='red', linestyle='--')
            y_max = max((*u_before.real, *u_before.imag, *u.real, *u.imag))
            plt.ylim(0, y_max)
            plt.xlim(-10, 10)
            # Plot Potential
            plt.plot(x, V, label="Potential $V(x)$", color='green', linestyle='--')

            # Add labels, title, and legend
            plt.title("$V(x)$, $u_{\mathrm{real}}(x,0)$ and $u_{\mathrm{imag}}(x,0)$")
            plt.xlabel("x")
            plt.legend()
            plt.grid()

            # Save and show the plot
            plt.savefig("newton_steady_state.svg")
            plt.close()
            system(f"rsvg-convert -f pdf -o newton_steady_state.pdf newton_steady_state.svg")
            system(f"open newton_steady_state.pdf")
#        print(max(u))

        return (True, u, x, dx, N, g, point_5_over_dx_squared, one_over_six) if return_all else (True, u)

    def refine_with_newton(plot_steady_state = False, return_all = True, xStart = None, lr=1):
        result = refine_with_newton_helper(plot_steady_state = plot_steady_state, return_all = return_all, xStart = torch.tensor(xStart) if xStart else None, lr=lr)
        while not result[0]:
            lr *= 0.5
            result = refine_with_newton_helper(plot_steady_state = plot_steady_state, return_all = return_all, xStart = torch.tensor(xStart) if xStart else None, lr=lr)
        return result[1:]

#    refine_with_newton(plot_steady_state=True)
#    exit()
    u, x, dx, N, g, point_5_over_dx_squared, one_over_six = refine_with_newton()

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
    global best_loss
    best_loss = np.inf
    global best_t_value
    best_t_value = T
    global df
    df = pd.DataFrame(columns=["x_0", "A", "b", "m", "Omega", "best_loss", "best_time"])
    try:
        df = pd.read_csv(f"{DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt", names=("x_0", "A", "b", "m", "Omega", "best_loss", "best_time"))
    except FileNotFoundError:
        # Create the file if it doesn't exist
        with open(f"{DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt", 'w') as f:
            if not no_print:
                print(f"File {DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt created successfully.")

    if not no_print:
        print(df)
    parameters = (df[(df['x_0']==x_start) & (df['A']==A) & (df['b']==b) & (df['m']==m) & (df['Omega']==Omega)]) if not df.empty else []
    closest_row_idx = -1
    closest_x, closest_A, closest_b, closest_m, closest_Omega = x_start, A, b, m, Omega

    if len(parameters):
        if not no_print:
            print(parameters)
        best_loss, best_t_value = parameters['best_loss'].item(), parameters['best_time'].item()
        try:
            t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
        except IndexError:
            try:
                t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
            except IndexError:
                t_test_values = t_values

        closest_row_idx = parameters.index[0]
    elif not df.empty:
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
    model_path = f"{DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_{flt_to_str(closest_x)}_{flt_to_str(round(closest_A, 2))}_{flt_to_str(round(closest_b, 2))}_{flt_to_str(round(closest_m, 2))}_{flt_to_str(round(closest_Omega, 2))}_pde_.pth" if not base_model else base_model
    new_model_path = f"{DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pth"
    if not no_print:
        print(f"model_path = {model_path}")
        print(f"new_model_path = {new_model_path}")
    if closest_x == x_start and round(closest_A, 2) == round(A, 2) and round(closest_b, 2) == round(b, 2) and round(closest_m, 2) == round(m, 2) and round(closest_Omega, 2) == round(Omega, 2):
        if not no_print:
            print("exact match found")
        assert(model_path == new_model_path or base_model)
    else:
        if not no_print:
            print(f"closest_x = {closest_x}, closest_A = {closest_A}, closest_b = {closest_b}, closest_m = {closest_m}, closest_Omega = {closest_Omega}")
            print(f"x_start = {x_start}, A = {A}, b = {b}, m = {m}, Omega = {Omega}")
    if useLibTorch:
        new_model_path = new_model_path.replace(".pth", ".pt")
    if os.path.exists(model_path) and load_model:
        if useLibTorch:
            model = torch.jit.load(model_path.replace(".pth", ".pt"))
        else:
            model.load_state_dict(torch.load(model_path))#, weights_only=True))
        if not no_print:
            print("Model loaded from file.")
            print(f"Example input of {example} yields: {model(example)[0, 0]}")
    elif os.path.exists(model_path) and not load_model:
        if not no_print:
            print("Model exists but not loaded.")
        if useLibTorch:
            new_model_path = new_model_path.replace(".pth", ".pt")
    else:
        if not no_print:
            print("No saved model found.")

    u = torch.tensor(u, dtype=torch.cfloat, requires_grad=True)
    density = u * torch.conj(u)  # Equivalent to |u|^2
    x = torch.tensor(x)

    # Calculate xmax (center of mass)
#    numerator = torch.sum(x * density.real) * dx
#    denominator = torch.sum(density.real) * dx
    numerator = torch.trapezoid(x * density.real, x)
    denominator = torch.trapezoid(density.real, x)
    xmax = numerator / denominator
    if not no_print:
        print(f"xmax = {xmax}")

    # Assuming u is a PyTorch tensor
    u_start = u.clone().detach().requires_grad_(True)
    if not no_print:
        print(f"len(u_start) = {len(u_start)}")

    def dxdt(v):
        return v

    def dvdt(x, xi):
        return force(x, xi) / m

    def NLS_RHS(u, N, g, V):
        """
        Compute the RHS of the NLS equation with periodic boundary conditions.

        Parameters:
            u (torch.Tensor): Solution array (complex tensor).
            N (int): Number of grid points.
            g (float): Nonlinearity coefficient.
            V (torch.Tensor): Potential array.

        Returns:
            torch.Tensor: Right-hand side of the NLS equation.
        """
        # Apply periodic boundary conditions
        up = torch.cat((u[-1:], u[:-1]))  # u(N) and u(1:N-1)
        um = torch.cat((u[1:], u[:1]))    # u(2:N) and u(1)

        # Compute the RHS
        RHS = 1j * (point_5_over_dx_squared * (up - 2 * u + um) - (g * u * torch.conj(u) + V) * u)

        return RHS

    def ODE_RK4(u, N, g, V, dt):
        """
        Perform one step of the Runge-Kutta 4th order (RK4) method for the NLS equation.

        Parameters:
            u (torch.Tensor): Current solution (complex tensor).
            N (int): Number of grid points.
            g (float): Nonlinearity coefficient.
            V (torch.Tensor): Potential array.
            dt (float): Time step.

        Returns:
            torch.Tensor: Updated solution after one RK4 step.
        """

        # Compute RK4 coefficients
        k1 = dt * NLS_RHS(u, N, g, V)
        k2 = dt * NLS_RHS(u + 0.5 * k1, N, g, V)
        k3 = dt * NLS_RHS(u + 0.5 * k2, N, g, V)
        k4 = dt * NLS_RHS(u + k3, N, g, V)

        # Update the solution
        RK4 = u + (k1 + 2 * k2 + 2 * k3 + k4) * one_over_six

        return RK4

    # Define the RHS function for solve_ivp
    def rhs_func(t, u_flat, V):
        u_flat = torch.tensor(u_flat, dtype=torch.complex64)
        rhs = NLS_RHS(u_flat, N, g, V)
        return rhs.detach().numpy().flatten()

    def ODE_DOP853(u, N, g, V, t, dt):
        """
        Perform one step of the DOP853 method for the NLS equation.

        Parameters:
            u (torch.Tensor): Current solution (complex tensor).
            N (int): Number of grid points.
            g (float): Nonlinearity coefficient.
            V (torch.Tensor): Potential array.
            dt (float): Time step.

        Returns:
            torch.Tensor: Updated solution after one DOP853 step.
        """
        # Convert torch.Tensor to numpy array for use with scipy
        u_np = u
        V_np = V

        # Time span for the ODE solver
        t_span = (t, t+dt)  # Solve from t=t to t=t+dt
        # Solve the ODE using DOP853
        sol = solve_ivp(rhs_func, t_span, u_np.flatten(), args=(V,), method='DOP853', t_eval=[t+dt])

        # Extract the solution at t = dt
        return sol.y

    # Define the xi function using the neural network
    def xi(t):
        t_input = torch.tensor([[t]], dtype=torch.float32)  # Convert to tensor
        return model(t_input)[0, 0]  # Get the output from the model

    def simulate_trajectory(xStart = None):
        nonlocal A, b, Omega, u, u_start, dt, x_start, v_start, add_kick
        xStart = x_start if not xStart else xStart
        x_values = [xStart]

        u, x, dx, N, g, point_5_over_dx_squared, one_over_six = refine_with_newton(xStart = torch.tensor([[xStart]], dtype=torch.float32), plot_steady_state=True)
        # Compute the density (modulus squared of u)
        density = u * np.conj(u)  # Equivalent to |u|^2
        # Calculate xmax (center of mass)
        numerator = np.trapz(x * density.real, x)
        denominator = np.trapz(density.real, x)
        xmax = numerator / denominator

        mass_values = [denominator] if simulate_only["store mass values"] else None # Store mass at each time step

        if not no_print:
            print(f"xStart = {xStart}, xmax = {xmax}")
        if add_kick:
            u *= np.exp(1j*v_start*x)
        u = torch.tensor(u, dtype=torch.cfloat)
        x = torch.tensor(x)
        x_flat = x
        for i in range(1, len(t_values)):
            V = potential(x, xi = 0)
            u = torch.tensor(ODE_DOP853(u, N, g, V, t_values[i], dt), dtype=torch.cfloat)

            # Compute the density (modulus squared of u)
            density_real_flat = (torch.conj(u)*u).real.flatten() # Equivalent to |u|^2
            # Calculate xmax (center of mass)
#            print(f"x.shape = {x.shape}, density.real.shape = {density.real.shape}, u.shape = {u.shape}")
            numerator = torch.trapezoid((x_flat * density_real_flat), x_flat)
            denominator = torch.trapezoid(density_real_flat, x_flat)
            x_values.append((numerator / denominator))
            if simulate_only["store mass values"]:
                mass_values.append(denominator)
        if simulate_only["store mass values"]:
            for i in range(1, len(mass_values)):
                mass_values[i] = 1 - mass_values[i]/mass_values[0]
            mass_values[0] = 0
        return x_values, t_values, xmax, mass_values

    if simulate_only["simulate_only"]:
        return simulate_trajectory(simulate_only["xStart"])

    def loss_func():
        nonlocal A, b, Omega, u, u_start, dt, x_start, v_start, x
        smoothness_penalty = 0.0  # Initialize smoothness penalty
        xi_values_temp = [torch.tensor([[0]], dtype=torch.float32)[0, 0]]  # Temporary storage for xi values to calculate smoothness
        v_values_temp = [v_start] # Temporary storage for v values to calculate max velocity
        best_loss_ = np.inf
        best_time = np.inf
        best_time_idx = np.inf
        x_values = [x_start]
        u = u_start.detach().requires_grad_(False)
#        print(f"x_start in loss_func first = {x_start}")

        for i in range(1, len(t_values)):
            xi_t = xi(t_values[i])
            xi_values_temp.append(xi_t)  # Store xi values for smoothness calculation
            V = potential(x, xi = xi_t)
            u = torch.tensor(ODE_DOP853(u, N, g, V, t_values[i], dt), dtype=torch.cfloat)

            # Compute the density (modulus squared of u)
            density_real_flat = (torch.conj(u)*u).real.flatten() # Equivalent to |u|^2

            # Calculate xmax (center of mass)
            numerator = torch.trapezoid(x * density_real_flat, x)
            denominator = torch.trapezoid(density_real_flat, x)
            x_values.append(numerator / denominator)
            #TODO: plot x_values vs t_values here once just to check for sanity's sake that it matches what I see in `compareTrajectories.py`
        v = (x_values[2] - x_values[0]) / (2 * dt)

        v_values_temp.append(abs(v))

        for i in range(2, len(x_values)):
            v = (x_values[i + 1] - x_values[i - 1]) / (2 * dt) if i < len(x_values) - 1 else ((x_values[-1] - x_values[-2]) / dt)
            v_values_temp.append(abs(v))

            x_star_x_diff = (x_star - x_values[i])
            x_star_xi_diff = (x_star - xi_values_temp[i])

            MSE = x_star_x_diff_mse_penalty * (x_star_x_diff*x_star_x_diff) + v_mse_penalty * (v*v) + x_star_xi_diff_mse_penalty * (x_star_xi_diff*x_star_xi_diff)

            if MSE < best_loss_:
                best_time = t_values[i]
                best_loss_ = MSE
                best_time_idx = i

        v_best = v_values_temp[0]
        xi_best = abs(xi_values_temp[0])
        for i in range(1, int(best_time_idx)+1):
            delta_xi = xi_values_temp[i] - xi_values_temp[i - 1]
            derivative = delta_xi / delta_t
            smoothness_penalty += derivative*derivative # Penalty based on the square of the "derivative"
            if v_values_temp[i] > v_best:
                v_best = v_values_temp[i]
            abs_xi_temp_i = abs(xi_values_temp[i])
            if abs_xi_temp_i > xi_best:
                xi_best = abs_xi_temp_i

        smoothness_penalty /= best_time_idx
        if not no_print:
            print(f"best_loss_ = {best_loss_}, smoothness_penalty = {smoothness_penalty},\nbest_time = {best_time}, v_best = {v_best:}\nabs_xi_temp_i = {xi_best}")
#        return (best_loss_ + smoothness_penalty_factor*smoothness_penalty + time_penalty_factor*best_time + velocity_penalty*v_best + xi_penalty*xi_best), best_time
        return (best_loss_ + smoothness_penalty_factor*smoothness_penalty + velocity_penalty*v_best + xi_penalty*xi_best), best_time


    def wrapped_loss_func(params):
        nonlocal A, b, Omega
        A, b, Omega = params
        return loss_func()[0].detach().numpy()

    # Loss function for optimization
    def closure():
        optimizer.zero_grad()  # Clear previous gradients
        loss, _ = loss_func()  # Compute the loss
        loss.backward()  # Backpropagate
        return loss

    # Define Newton's method function
    def newton_method():
        global best_loss, df, t_test_values, best_t_value
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
                    if not no_print:
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
                    if not no_print:
                        print("libtorch version saved")
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A,2), 'b': round(b,2), 'm': round(m,2), 'Omega': round(Omega,2), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    if not no_print:
                        print(f"New best t value = {best_t_value}")
                try:
                    t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                except:
                    t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                # Write to CSV
                df.to_csv(f'{DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt', header=None, index=False)
                if not no_print:
                    print(f"Best model saved with loss: {best_loss}")
            else:
                # Revert to previous parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(current_params[name])

            # Log current loss and time
            if not no_print:
                print(f"Curr Loss = {new_loss}, Curr Time = {new_time:.6f}")

        # Restore best parameters to the model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(best_params[name])

    # Define brute force function
    def brute_force(fine = False, coolingRate = 0.99, anneal = False, initial_temp = 1):
        global best_loss, df, t_test_values, best_t_value
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
                    if not no_print:
                        print("libtorch version saved")
                df = add_or_update_row(df, {'x_0': x_start, 'A': round(A,2), 'b': round(b,2), 'm': round(m,2), 'Omega': round(Omega,2), 'best_loss': best_loss.detach().numpy(), 'best_time': best_t_value})
                if best_t_value != t_test_values[-1]:
                    if not no_print:
                        print(f"New best t value = {best_t_value}")
                try:
                    t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                except:
                    t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                # Write to CSV
                df.to_csv(f'{DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt', header=None, index=False)
                if not no_print:
                    print(f"Best model saved with loss: {best_loss}")

            elif anneal and torch.exp(-delta_loss / temperature) > np.random.random():
                current_params = perturbed_params
            else:
                # Revert to previous parameters
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        param.copy_(current_params[name])

            temperature *= cooling_rate
            if not no_print:
              print(f"Curr Loss = {new_loss:.6f}, Curr Time = {new_time:.6f}, Curr Temp = {temperature:.6e}")

        # Restore best parameters to the model
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(best_params[name])

    loss_value_test = loss_func()
    if not no_print:
        print(f"closest_x = {closest_x}, closest_A = {closest_A}, closest_b = {closest_b}, closest_m = {closest_m}, closest_Omega = {closest_Omega}")
        print(f"x_start = {x_start}, A = {A}, b = {b}, m = {m}, Omega = {Omega}")
        print("best_loss =", best_loss)
        print("Current loss and time =", loss_value_test)

    if get_model_loss_value:
        if optimize_A_b_Omega_m:
#            initial_guess = [A, b, Omega]
#            loss_value_test = minimize(wrapped_loss_func, initial_guess)
#            return [loss_value_test.fun, *loss_value_test.x]
            # Hyperparameters
            num_particles = 10
            num_iterations = optimize_A_b_Omega_m_iterations
            w = 5   # Inertia weight
            c1 = 15  # Cognitive component
            c2 = 15  # Social component

            # Initialize particles around the current values
            particles = [
                {
                    "position": {
                        "A": A + 0.01 * np.random.randn(),
                        "b": b + 0.01 * np.random.randn(),
                        "Omega": Omega + 0.01 * np.random.randn(),
                    },
                    "velocity": {
                        "A": 0.0,
                        "b": 0.0,
                        "Omega": 0.0,
                    },
                    "best_position": None,
                    "best_value": float("inf"),
                }
                for _ in range(num_particles)
            ]

            # Initialize global best
            global_best_position = None
            u = torch.tensor(refine_with_newton(return_all = False), dtype=torch.cfloat, requires_grad=True)
            u_start = u.clone().detach().requires_grad_(True)
            global_best_value = loss_value_test[0].detach().numpy()
            print(f"Starting loss = {global_best_value}")
            print(f"Starting position: A = {A}, b = {b}, Omega = {Omega}")

            try:
                for i in (range(num_iterations) if num_iterations != np.inf else iter(int, 1)):
                    for particle in particles:
                        # Update global variables with particle values
                        A, b, Omega = (
                            particle["position"]["A"],
                            particle["position"]["b"],
                            particle["position"]["Omega"],
                        )

                        # Evaluate loss function
                        x_start, v_start = get_x_start_and_v_start()
                        epsilon = 1e-1
                        while np.abs(x_start) < min_x_start:
                            # Perturb A, b, Omega by a small amount
                            perturbation_A = np.random.normal(0, epsilon)  # Small random perturbation for A
                            perturbation_b = np.random.normal(0, epsilon)  # Small random perturbation for b
                            perturbation_Omega = np.random.normal(0, epsilon)  # Small random perturbation for Omega

                            A = A + perturbation_A
                            b = b + perturbation_b
                            Omega = Omega + perturbation_Omega
                            x_start, v_start = get_x_start_and_v_start()


                        u = refine_with_newton(return_all = False)

                        u_start = torch.tensor(u, dtype=torch.cfloat, requires_grad=True).clone().detach()
                        loss = loss_func()[0].detach().numpy()

                        # Update personal best
                        if loss < particle["best_value"]:
                            particle["best_value"] = loss
                            particle["best_position"] = particle["position"].copy()

                        # Update global best
                        if loss < global_best_value:
                            global_best_value = loss
                            global_best_position = particle["position"].copy()
                            print(f"New best loss = {global_best_value}")
                            print(f"New best position: A = {global_best_position['A']}, b = {global_best_position['b']}, Omega = {global_best_position['Omega']}")

                    # Update velocities and positions
                    for particle in particles:
                        for key in ["A", "b", "Omega"]:
                            r1, r2 = np.random.rand(), np.random.rand()
                            cognitive = c1 * r1 * (particle["best_position"][key] - particle["position"][key])
                            social = c2 * r2 * (global_best_position[key] - particle["position"][key])
                            particle["velocity"][key] = w * particle["velocity"][key] + cognitive + social
                            particle["position"][key] = np.abs(particle["position"][key] + particle["velocity"][key])
            except KeyboardInterrupt:
                # Apply the best found solution
                A, b, Omega = global_best_position["A"], global_best_position["b"], global_best_position["Omega"]
                return loss_func()[0].detach().numpy(), A, b, Omega
        return loss_value_test

    if not automate:
        ans = input("Proceed? (y/n): ")
        if ans.lower() != 'y':
            exit()

    # Training loop
    global learning_rate
    learning_rate = 0.001
    Algorithm = "brute force"
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
            brute_force(fine = False, coolingRate = 0.99, anneal = False, initial_temp = .03)
            if raiseBaseException:
                raise(KeyboardInterrupt)
        elif Algorithm == "newton":
            newton_method()
            if raiseBaseException:
                raise(KeyboardInterrupt)
        else:
            while criterion():
#                if epoch >= 1000:
#                    return best_loss
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
                        if not no_print:
                            print("libtorch version saved")
                    df = add_or_update_row(df, {'x_0': x_start, 'A': A, 'b': b, 'm': m, 'Omega': Omega, 'best_loss': best_loss, 'best_time': best_t_value})
                    if best_t_value != t_test_values[-1]:
                        if not no_print:
                            print(f"New best t value = {best_t_value}")
                        try:
                            t_test_values = t_values[:np.where(t_values==best_t_value)[0][0]+1]
                        except:
                            t_test_values = t_values[:np.where(np.isclose(t_values, best_t_value))[0][0]+1]
                    # Write to CSV
                    df.to_csv(f'{DRIVE_BASE_PATH}/dataFiles/ICsPDE.txt', header=None, index=False)
                    if not no_print:
                        print(f"Best model saved with loss: {best_loss}")

                loss_value.backward()  # Compute gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        #        optimizer.step()  # Update weights
                if Algorithm == "lbfgs":
                    optimizer.step(closure)
                else:
                    optimizer.step()
                if not no_print:
                    print(f'Epoch {epoch}, Loss: {loss_value.item()}')

                if plot_progress and epoch % 5 == 0:
                    x_values, v_values, xi_values, a_values = [x_start], [v_start], [0.0], [0.0]
                    u = u_start.clone()
                    for i in range(1, len(t_test_values)):
                        xi_t = xi(t_values[i])
                        xi_values.append(xi_t.detach().numpy())
                        V = potential(x = x_values[-1], xi = xi_values[-1])
                        u = ODE_RK4(u, N, g, V, dt)

                        # Compute the density (modulus squared of u)
                        density = u * torch.conj(u)  # Equivalent to |u|^2

                        # Calculate xmax (center of mass)
#                        numerator = torch.sum(x * density.real)
#                        denominator = torch.sum(density.real)
                        numerator = torch.trapezoid(x * density.real, x)
                        denominator = torch.trapezoid(density.real, x)
                        x_values.append((numerator / denominator).detach().numpy())

                    v = (x_values[2] - x_values[0]) / (2 * dt)
                    v_values.append(v)
                    for i in range(2, len(x_values)):
                        v = (x_values[i + 1] - x_values[i - 1]) / (2 * dt) if i < len(x_values) - 1 else ((x_values[-1] - x_values[-2]) / dt)
                        v_values.append(v)

                    a = (v_values[2] - v_values[0]) / (2 * dt)
                    a_values.append(a)
                    for i in range(2, len(v_values)):
                        v = (v_values[i + 1] - v_values[i - 1]) / (2 * dt) if i < len(v_values) - 1 else ((v_values[-1] - v_values[-2]) / dt)
                        a_values.append(v)


                    plt.plot(t_test_values, x_values, label='x(t) [m]', color='blue')
                    plt.plot(t_test_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
                    plt.plot(t_test_values, xi_values, label=r'$\xi(t)$', color='red', linestyle='--')
                    plt.plot(t_test_values, a_values, label='a(t) [m/$s^2$]', color='purple', linestyle='-.')

                    plt.legend()
                    plt.draw()
                    plt.pause(1)
                    plt.close()
                epoch += 1
            if raiseBaseException:
                raise(KeyboardInterrupt)

    except KeyboardInterrupt:
        if not no_print:
            print("\nTraining interrupted. Saving data...")

        # Save the best model if it was updated
        if best_loss < float('inf'):  # Ensure that at least one model has been saved
            if not no_print:
                print(f"Best model was saved with loss: {best_loss}")
        else:
            if not no_print:
                print("No new best model found.")
            exit()

        model = XiModel()
        if not no_print:
            print("New model path loaded =", new_model_path)
        if useLibTorch and os.path.exists(new_model_path):
            if not no_print:
                print(f"new_model_path = {new_model_path}")
            model = torch.jit.load(new_model_path)
        else:
            model.load_state_dict(torch.load(new_model_path))#, weights_only=True))
        xi = lambda t: model(torch.tensor([[t]], dtype=torch.float32))[0, 0] # Get the output from the model
        # Save data to CSV
        data_path = f"{DRIVE_BASE_PATH}/dataFiles/trajectory_data_pde.csv"
        x_values, v_values, xi_values, a_values, u_values, densities = [x_start], [v_start], [0.0], [0.0], [u_start.clone()], [np.squeeze(u_start.detach().numpy() * np.conj(u_start.detach().numpy()))]
        for i in range(1, len(t_test_values)):
            xi_t = xi(t_values[i])
            xi_values.append(xi_t.detach().numpy())
            V = potential(x, xi = xi_values[-1])
            u_values.append(ODE_DOP853(u_values[-1].detach().numpy() if isinstance(u_values[-1], torch.Tensor) else u_values[-1], N, g, V, t_test_values[i], dt))
            # Compute the density (modulus squared of u)
            density = np.squeeze(u_values[-1] * np.conj(u_values[-1]))  # Equivalent to |u|^2
            #print(f"u_values[-1].shape = {u_values[-1].shape}, x.shape = {x.shape}, density.shape = {density.shape}")

            # Calculate xmax (center of mass)
            numerator = np.trapz(x * density.real, x)
            denominator = np.trapz(density.real, x)
            x_values.append(numerator / denominator)
            densities.append(density)

        v = (x_values[2] - x_values[0]) / (2 * dt)
        v_values.append(v)
        for i in range(2, len(x_values)):
            v = (x_values[i + 1] - x_values[i - 1]) / (2 * dt) if i < len(x_values) - 1 else ((x_values[-1] - x_values[-2]) / dt)
            v_values.append(v)

        a = (v_values[2] - v_values[0]) / (2 * dt)
        a_values.append(a)
        for i in range(2, len(v_values)):
            v = (v_values[i + 1] - v_values[i - 1]) / (2 * dt) if i < len(v_values) - 1 else ((v_values[-1] - v_values[-2]) / dt)
            a_values.append(v)

        with open(data_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["t_values", "xi_values", "x_values", "v_values"])
            for i in range(len(t_test_values)):
                writer.writerow([t_values[i], xi_values[i], x_values[i], v_values[i]])
        if not no_print:
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
        plt.savefig("trajectory_data_pde.svg")
        system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf trajectory_data_pde.svg")
        system(f"open trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf")
        system(f"cp trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
        system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.png")
        system(f"open {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.png")
        if not no_print:
            print(f"image {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.png saved")
        system(f"rm trajectory_data_pde.svg trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf ")
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
            plt.savefig("trajectory_data_pde.svg")
            system(f"rsvg-convert -f pdf -o trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf trajectory_data_pde.svg")
            system(f"open trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf")
            system(f"cp trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/")
            system(f"sips -s format png -s dpiWidth 480 -s dpiHeight 480 -z 2400 2400 {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf --out ../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.png")
            system(f"open {DRIVE_BASE_PATH}/imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.png")
            system(f"rm trajectory_data_pde.svg trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.pdf")

            plt.close()

        if not automate:
            answer = input("Movie (y/n)? ")
            if not answer.lower().startswith('y'):
                if not no_print:
                    print("Movie creation skipped.")
                exit()

        N = len(t_test_values)
        x_range = np.linspace(-10, 10, len(u_start))
        fig, ax = plt.subplots(figsize=(8, 6))
        # Initialize plot elements
        dot, = ax.plot([], [], 'r', lw=1, label = "Soliton")
        center, = ax.plot([], [], 'o', color='orange', markersize=8, label="Center of Mass")
        curve, = ax.plot([], [], 'b-', lw=2)
    #    gold_dot, = ax.plot([], [], 'yo', markersize=8, label = "$x^*$")
        # Set plot limits and labels
        ax.set_xlim(x_range[0], x_range[-1])
        ax.set_ylim(0, 5)
        ax.set_xlabel("x")
        ax.set_ylabel("Potential")
        ax.legend()

        ax.set_title(f'$x_0$ = {x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
        sech = lambda x: 1/np.cosh(x)

        def potential(x, xi):
            nonlocal A, b, Omega
            V_MT = 0.5 * Omega * Omega * x * x
            temp = sech(b * (x - xi))
            V_SECH = A * temp * temp
            return V_MT + V_SECH
        def potential_magnetic_trap(x, xi):
            nonlocal A, b, Omega
            V_MT = 0.5 * Omega * Omega * x * x
            return V_MT
        # Initialization function
        def init():
    #        gold_dot.set_data([], [])
            dot.set_data([], [])
            center.set_data([], [])
            curve.set_data([], [])
            return dot, curve, center#, gold_dot

            # Update function
        def update(i):
            t = (i) * (best_t_value / (N - 1))  # Calculate current time
            y_values = potential(x_range, xi_values[i])
#            u_np = u_values[-1].detach().numpy() if isinstance(u_values[-1], torch.Tensor) else u_values[-1]
#            density = np.squeeze(u_np * np.conj(u_np))
            dot.set_data(x, densities[i])
            center.set_data([x_values[i]], [potential(x_values[i], xi_values[i])])
            curve.set_data(x_range, y_values)
    #        gold_dot.set_data([x_star], [potential(x_star, xi_values[i])])
            ax.set_title(f"$x_0$ = {x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
            return dot, curve, center#, gold_dot

        # Create animation
        fps = N/best_t_value
        if not no_print:
#            print(xi_values, type(xi_values))
            print(f"Omega = {Omega}, b = {b}, A = {A}, xi_0 = {xi_values[0]}")

        ani = animation.FuncAnimation(fig, update, frames=range(0, N, 1), init_func=init, blit=True, interval = 1000/fps)

        # Save the animation
        ani.save(f"{DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4", writer=animation.FFMpegWriter(fps=2*fps))
        system(f"open {DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4")
        if not no_print:
            print(f"Movie saved as '{DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4'")
        plt.close()

        if produceInverse:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Initialize plot elements
            dot, = ax.plot([], [], 'r', lw=1, label = "Soliton")
            center, = ax.plot([], [], 'o', color='orange', markersize=8, label="Center of Mass")
            curve, = ax.plot([], [], 'b-', lw=2)
    #        gold_dot, = ax.plot([], [], 'yo', markersize=8, label = "$x^*$")

            # Set plot limits and labels
            ax.set_xlim(x_range[0], x_range[-1])
            ax.set_ylim(0, 5)
            ax.set_xlabel("x")
            ax.set_ylabel("Potential")
            ax.legend()
            ax.set_title(f'$x_0$ = {-x_start:.2f}, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}')
            sech = lambda x: 1/np.cosh(x)
            def update(i):
                t = (i) * (best_t_value / (N - 1))  # Calculate current time
                y_values = potential(x_range, -xi_values[i])
                density = (u_values[i] * torch.conj(u_values[i])).detach().numpy()[::-1]
                dot.set_data(x, density + potential_magnetic_trap(x_range, xi_values[i]))
                center.set_data([-x_values[i]], [potential(x_values[i], -xi_values[i])])
                curve.set_data(x_range, y_values)
        #        gold_dot.set_data([x_star], [potential(x_star, xi_values[i])])
                ax.set_title(f"$x_0$ = {x_values[0]:.2f}, $x^*$ = 0, $A$ = {A:.2f}, $b$ = {b:.2f}, $m$ = {m:.2f}, $\Omega$ = {Omega:.2f}, t = {t:.2f}")
                return dot, curve, center#, gold_dot
            ani = animation.FuncAnimation(fig, update, frames=range(0, N, 10), init_func=init, blit=True, interval = 1000/fps)

            # Save the animation
            ani.save(f"{DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4", writer=animation.FFMpegWriter(fps=2*fps))
            system(f"open {DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4")
            if not no_print:
                print(f"Movie saved as '{DRIVE_BASE_PATH}/movies/trajectory_trap_plus_sech_squared/trajectory_data_IC_{flt_to_str(-x_start)}_{flt_to_str(round(A, 2))}_{flt_to_str(round(b, 2))}_{flt_to_str(round(m, 2))}_{flt_to_str(round(Omega, 2))}_pde_.mp4'")

        plt.close()

def check_losses_on_pde_for_learned_odes(file_choice = "all"):
    loss_file_pairs = []
    if file_choice == "all":
        for model_file in glob("{DRIVE_BASE_PATH}/NeuralNetworkData/*pth"):
            try:
                loss = master_func_learn_ivp_pde(load_model = True, base_model = model_file, no_print = True, get_model_loss_value = True)[0].item()
                print(f"model_file = {model_file}, loss = {loss}")
                loss_file_pairs.append((model_file, loss))
            except Exception as e:
                print(f"Error for file {model_file}: {e}")
                print("Continuing.")
    else:
#        try:
        loss = master_func_learn_ivp_pde(load_model = True, base_model = file_choice, no_print = True, get_model_loss_value = True)[0].item()
        print(f"model_file = {file_choice}, loss = {loss}")
        loss_file_pairs.append((file_choice, loss))
#        except Exception as e:
#            print(f"Error for file {file_choice}: {e}")
#            print("Continuing.")

    print(*loss_file_pairs, sep='\n')
    best_file, best_loss = min(loss_file_pairs, key = lambda x: x[1])
    worst_file, worst_loss = max(loss_file_pairs, key = lambda x: x[1])
    print("="*20)
    print(f"Best loss = {best_loss}")
    print(f"Best file = {best_file}")
    print(f"Worst loss = {worst_loss}")
    print(f"Worst file = {worst_file}")
#    Best loss = 4.934653040667099
#    Best file = {DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_2_point_2258_1_point_0_1_point_0_1_point_3_0_point_2_.pth
#    Worst loss = 5.5253880920493375
#    Worst file = {DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_2_point_4513_1_point_7_1_point_0_1_point_0_0_point_2_.pth

def optimize_losses_on_pde_for_learned_odes(file_choice = "all"):
    if file_choice == "all":
        loss_file_pairs = []
        for model_file in glob("{DRIVE_BASE_PATH}/NeuralNetworkData/*pth"):
    #        try:
            loss, optim_A, optim_b, optim_Omega = master_func_learn_ivp_pde(load_model = True, base_model = model_file, no_print = False, get_model_loss_value = True, optimize_A_b_Omega_m = True)
            print(f"model_file = {model_file}, loss = {loss}, A = {optim_A}, b = {optim_b}, Omega = {optim_Omega}")
            loss_file_pairs.append((model_file, loss, optim_A, optim_b, optim_Omega))
    #        except Exception as e:
    #            print(f"Error for file {model_file}: {e}")
    #            print("Continuing.")

        print(*loss_file_pairs, sep='\n')
        best_file, best_loss = min(loss_file_pairs, key = lambda x: x[1])
        print(f"Best loss = {best_loss}")
        print(f"Best file = {best_file}")
    else:
        #BEST SO FAR
        #Starting loss = 0.8616801642329329, Starting position: A = 1.287276611039334, b = 2.852755931077745, Omega = 0.37507858278618567
        loss, optim_A, optim_b, optim_Omega = master_func_learn_ivp_pde(load_model = True, base_model = file_choice, no_print = True, get_model_loss_value = True, optimize_A_b_Omega_m = True, A = 1.2922437525694463, b = 2.851373288066033, Omega = 0.3815554681671926)
        print(f"file_choice = {file_choice}, loss = {loss}, A = {optim_A}, b = {optim_b}, Omega = {optim_Omega}")

if __name__ == "__main__":
#    optimize_losses_on_pde_for_learned_odes(file_choice = f"{DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_2_point_2258_1_point_0_1_point_0_1_point_3_0_point_2_.pth")
#    check_losses_on_pde_for_learned_odes(file_choice="xi_model_IC_2_point_4063_0_point_66_0_point_75_1_point_0_0_point_2_.pt")
#    master_func_learn_ivp_pde(load_model = True, base_model = "xi_model_IC_2_point_5715_0_point_68_0_point_67_1_point_0_0_point_2_.pt", T = 10)
    master_func_learn_ivp_pde(load_model = True, T = 10, A = 0.75, b = 1, Omega = .2)
#    master_func_learn_ivp_pde(load_model = True, A = 0.1, b = 1, base_model = "xi_model_IC_0_point_787127_1_0_point_1_1_point_0_0_point_2_Paul_.pt", T = 10)
#    master_func_learn_ivp_pde(load_model = True, A = 0.1, b = 1, base_model = "xi_model_IC_0_point_848359_0_point_067475_0_point_665792_1_point_0_0_point_2_.pt", T = 10)

#    master_func_learn_ivp_pde(load_model = True)

    #master_func_learn_ivp_pde(load_model = True, base_model = f"{DRIVE_BASE_PATH}/NeuralNetworkData/xi_model_IC_0_point_9432_1_point_29_2_point_85_1_point_0_0_point_38_pde_.pth", no_print = False, A = 1.287276611039334, b = 2.852755931077745, Omega = 0.37507858278618567)
    exit()
    for A in np.arange(1.1, 2.1, 0.1):
        master_func_learn_ivp_pde(A=round(A,2))
    #    start = time()
    #    result = master_func_learn_ivp_pde(A=A)
    #    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
    #    with open("result_times.txt", "a") as f:
    #        f.write(f"Time from A = {A-0.1:.2f} to {A:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
    for m in np.arange(1.1, 2.1, 0.1):
        master_func_learn_ivp_pde(m=round(m,2))
    #    start = time()
    #    result = master_func_learn_ivp_pde(m=m)
    #    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
    #    with open("result_times.txt", "a") as f:
    #        f.write(f"Time from m = {m-0.1:.2f} to {m:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
    for b in np.arange(1., 2.1, 0.1):
        master_func_learn_ivp_pde(b=round(b,2))
    #    start = time()
    #    result = master_func_learn_ivp_pde(b=b)
    #    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
    #    with open("result_times.txt", "a") as f:
    #        f.write(f"Time from b = {b-0.1:.2f} to {b:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")
    for Omega in np.arange(0.3, 1.3, 0.1):
        master_func_learn_ivp_pde(Omega=round(Omega,2))
    #    start = time()
    #    result = master_func_learn_ivp_pde(Omega=Omega)
    #    achieved = 'target achieved' if result is None else f'target not achieved, best loss after epoch 1000 = {result}'
    #    with open("result_times.txt", "a") as f:
    #        f.write(f"Time from Omega = {Omega-0.1:.2f} to {Omega:.2f} = {time() - start:.2f} with learning rate {learning_rate}, {achieved}\n")




    #../imgs/pdfs/trajectory_pdfs_trap_plus_sech_squared/
    #../movies/trajectory_trap_plus_sech_squared/
    #/Users/edwardfinkelstein/RCPDE/EdwardF/scripts/result_times.txt


