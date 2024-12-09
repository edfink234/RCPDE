import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, tanh, arccos, sqrt, log as ln, arcsin, log, arcsin as asin, arccos as acos, exp
from matplotlib import rcParams
from sympy.parsing.sympy_parser import parse_expr
from sympy.printing.latex import latex

# Constants for the potential
m = 1.0        # Mass
Omega = 1.0    # Frequency of the harmonic trap
A = 1.0        # Amplitude of the Gaussian potential
sigma = 1.0    # Width of the Gaussian potential
T = 10.0       # Final time
dt = 0.01      # Time step
x_star = 0.5   # Final position sought
v_th = 0.01    # Velocity threshold

def expression_to_latex(expression_str):
    from sympy import symbols, sin, cos, tanh, sqrt, ln, acos, asin, acos as arccos, asin as arcsin
    try:
        # Define symbols used in the expression
        t = symbols('t')
        
        # Parse the input expression
        expr = parse_expr(expression_str, evaluate=False)
        
        # Convert to LaTeX
        latex_expr = latex(expr)
        from numpy import sin, cos, tanh, arccos, sqrt, log as ln, arcsin, log, arcsin as asin, arccos as acos, exp
        return latex_expr
    except Exception as e:
        return f"Error: {e}"

sech = lambda x: 1/np.cosh(x)
xi_val = "(((tanh(2) / 2) / 0.860789) * (cos(-((2 - 0.847350))) ** (sech(sin(((2 * t) / 2))) / t)))"
# Define the symbolic regression solution for xi(t)
def xi(t):
    return eval(xi_val)

# Function to compute the force (negative derivative of potential)
def force(x, xi):
#    return -(Omega**2 * (x - xi)) - (2.0 * A * (x - xi) / sigma) * np.exp(-(x - xi)**2 / sigma)
    return -(Omega**2 * x) + (2 * A**3 * sech(A * (x - xi))**2 * tanh(A * (x - xi)))

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

# Initial conditions

# Initial position x(0)
x_0 = -1.5
# Initial velocity v(0)
v_0 = 0.0

x = x_0
v = v_0

# Time evolution loop
for t in t_values:
    xi_t = xi(t)  # Compute xi(t) at time t
    xi_values.append(xi_t)
    
    # Perform RK4 step
    x, v = rk4_step(x, v, xi_t, dt)
    
    # Store the position x(t) and velocity v(t)
    x_values.append(x)
    v_values.append(v)

# Plot the results
#point_sz = 0.1 * (rcParams['lines.markersize'] ** 2)
plt.plot(t_values, x_values, label='x(t) [m]', color='blue')
plt.plot(t_values, v_values, label='v(t) [m/s]', color='green', linestyle=':')
#plt.scatter(t_values, xi_values, label = r'$\xi(t) = \xi(t) = \mathrm{cos}\left(\mathrm{sech}(1.165323) \cdot t\right) \cdot \mathrm{exp}\left(\frac{t}{-(10 + t)}\right)$', color='red', linestyle='--', s=point_sz)
plt.plot(t_values, xi_values, label = rf'$\xi(t) = {expression_to_latex(xi_val)}$', color='red', linestyle='--')

plt.xlabel('Time (s)')
plt.grid(True)
plt.legend()
plt.savefig("InvertedPendulumTrial.png", dpi=5*96)
import os
os.system("open InvertedPendulumTrial.png")
