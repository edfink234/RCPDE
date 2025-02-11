fit = False
from numpy import exp
from numpy import linspace
import numpy as np
import matplotlib.pyplot as plt
from os import system
if fit:
    from pysr import PySRRegressor
from sys import maxsize
from scipy.optimize import curve_fit

class Dum():
    def predict(self):
        pass

A = 1
rho = 0
Omega = 0.2
B = 1

Veff_no_trap = lambda n: (-(-0.16e2  *  B  *  A  *  (A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  n + A  * \
    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  n - A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  n - A  *  \
    exp((2  *  A  *  (4  *  n + rho)))  *  n - A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  rho - A  *  \
    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  rho + A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  rho + A  *  \
    exp((2  *  A  *  (4  *  n + rho)))  *  rho + 0.3e1  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.3e1  *  \
    exp((2  *  A  *  (3  *  n + 2  *  rho))) - exp((2  *  A  *  (n + 4  *  rho))) + \
    exp((2  *  A  *  (4  *  n + rho))))  /  (exp((10  *  A  *  rho)) - 0.5e1  *  \
    exp((2  *  A  *  (n + 4  *  rho))) + 0.10e2  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.10e2  *  \
    exp((2  *  A  *  (3  *  n + 2  *  rho))) + 0.5e1  *  exp((2  *  A  *  (4  *  n + rho))) - exp((10  *  A  *  n))))) / (2 * A);
    
suspicious_bs = lambda n: (exp((10  *  A  *  rho)) - 0.5e1  *  \
    exp((2  *  A  *  (n + 4  *  rho))) + 0.10e2  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.10e2  *  \
    exp((2  *  A  *  (3  *  n + 2  *  rho))) + 0.5e1  *  exp((2  *  A  *  (4  *  n + rho))) - exp((10  *  A  *  n)))

window=0.5
plt.plot(np.linspace(-window, window, 1000), suspicious_bs(np.linspace(-window, window, 1000)))
plt.ylim(-1,2)
plt.show()
plt.close()
exit()

def sech_fit(x, A, b):
    return A * sech(b * x)

def sech2_fit(x, A, b):
    return A * sech(b * x) ** 2

sech = lambda x: 1/np.cosh(x)

# Generate data
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = Veff_no_trap(x)

model=Dum()

if fit:
    # Fit using PySR until manually stopped
    model = PySRRegressor(
        niterations=1000,  # Run indefinitely until Ctrl+C
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin", "cos", "sech"],
        progress=True,
    )
    try:
        model.fit(x, y)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Displaying best model found so far:\n")
        print(model)

# Fit sech function
params_sech, _ = curve_fit(sech_fit, x.ravel(), y.ravel(), p0=[1, 1])
A_sech, b_sech = params_sech

# Fit sech^2 function
params_sech2, _ = curve_fit(sech2_fit, x.ravel(), y.ravel(), p0=[1, 1])
A_sech2, b_sech2 = params_sech2

# Generate fitted curves
y_fit_sech = sech_fit(x, A_sech, b_sech)
y_fit_sech2 = sech2_fit(x, A_sech2, b_sech2)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(x, y, label="Original", color="black")
model_func = lambda x: np.sin(np.sin((np.sin(sech(0.45118 * x)) - 0.02943) * sech(x)))
model.predict = model_func

# Compute MSE
mse_pysr = np.mean((y - model.predict(x))**2)
mse_sech = np.mean((y - y_fit_sech)**2)
mse_sech2 = np.mean((y - y_fit_sech2)**2)

plt.plot(x, model.predict(x), "--", label=r"$V(x) = \mathrm{sin}(\mathrm{sin}((\mathrm{sin}(\mathrm{sech}(0.45118 \cdot x)) - 0.02943) \cdot \mathrm{sech}(x)))$"f', MSE = {mse_pysr:.3e}', color="red")
plt.plot(x, y_fit_sech, label=r'$V(x) = A\cdot\mathrm{sech}(bx), $'f'$A={A_sech:.3f}$, $b={b_sech:.3f}$'f', MSE = {mse_sech:.3e}', linestyle='dashed')
plt.plot(x, y_fit_sech2, label=r'$V(x) = A\cdot\mathrm{sech}^2(bx), $'f'$A={A_sech2:.3f}, b={b_sech2:.3f}$'f', MSE = {mse_sech2:.3e}', linestyle='dotted')
plt.legend()
plt.xlabel("x")
plt.ylabel("V(x)")
#$\mathrm{sin}(\mathrm{sin}((\mathrm{sin}(\mathrm{sech}(0.45118 \cdot x)) - 0.02943) \cdot \mathrm{sech}(x)))$
plt.tight_layout()
plt.savefig("PaulVariationalPotentialPlot.png",dpi=5*96)
system("open PaulVariationalPotentialPlot.png")

