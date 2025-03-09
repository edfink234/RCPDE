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
B = 0.1
L_tay = -0.01
R_tay = 0.01

#Veff_no_trap = lambda n: (-(-0.16e2  *  B  *  A  *  (A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  n + A  * \
#    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  n - A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  n - A  *  \
#    exp((2  *  A  *  (4  *  n + rho)))  *  n - A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  rho - A  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  rho + A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  rho + A  *  \
#    exp((2  *  A  *  (4  *  n + rho)))  *  rho + 0.3e1  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.3e1  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho))) - exp((2  *  A  *  (n + 4  *  rho))) + \
#    exp((2  *  A  *  (4  *  n + rho))))  /  (exp((10  *  A  *  rho)) - 0.5e1  *  \
#    exp((2  *  A  *  (n + 4  *  rho))) + 0.10e2  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.10e2  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho))) + 0.5e1  *  exp((2  *  A  *  (4  *  n + rho))) - exp((10  *  A  *  n))))) / (2 * A);
#    
#suspicious_bs = lambda n: (exp((10  *  A  *  rho)) - 0.5e1  *  \
#    exp((2  *  A  *  (n + 4  *  rho))) + 0.10e2  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.10e2  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho))) + 0.5e1  *  exp((2  *  A  *  (4  *  n + rho))) - exp((10  *  A  *  n)))
#    
#numerator_suspicios_bs = lambda n: (A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  n + A  * \
#    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  n - A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  n - A  *  \
#    exp((2  *  A  *  (4  *  n + rho)))  *  n - A  *  exp((2  *  A  *  (2  *  n + 3  *  rho)))  *  rho - A  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho)))  *  rho + A  *  exp((2  *  A  *  (n + 4  *  rho)))  *  rho + A  *  \
#    exp((2  *  A  *  (4  *  n + rho)))  *  rho + 0.3e1  *  exp((2  *  A  *  (2  *  n + 3  *  rho))) - 0.3e1  *  \
#    exp((2  *  A  *  (3  *  n + 2  *  rho))) - exp((2  *  A  *  (n + 4  *  rho))) + \
#    exp((2  *  A  *  (4  *  n + rho))))

# Define the effective potential functions
#$V_{\text{eff}}(x) = \frac{8B\left(Ae^{4A(x-\xi)}(x-\xi)+Ae^{2A(x-\xi)}(x-\xi)-e^{4A(x-\xi)}+e^{2A(x-\xi)}\right)}{e^{6A(x-\xi)}-3e^{4A(x-\xi)}+3e^{2A(x-\xi)}-1}$
def Veff(x, xi = 0):
    temp = x-xi
    temp_2_exp = np.exp(2 * A * temp)
    temp_4_exp = np.exp(4 * A * temp)
    b = B
    return (8 * b * (A * temp_4_exp * temp + A * temp_2_exp * temp - temp_4_exp + temp_2_exp) /
        (np.exp(6 * A * temp) - 3 * temp_4_exp + 3 * temp_2_exp - 1))

#$V_{\text{eff\_tay}}(x) = - \frac{4}{15} B \left( A^2 (x-\xi)^2 - \frac{5}{2} \right)$

def Veff_tay(x, xi = 0):
    temp=x-xi
    return - 4 / 15 * (B) * (A*A * (temp*temp) - 5 / 2)

# Define the effective potential function that switches between Veff and Veff_tay
def Veff_no_trap(x, L, R, xi = 0):
    effpot = np.zeros_like(x)
    for i in range(len(x)):
        if L <= x[i] <= R:
            effpot[i] = Veff_tay(x[i], xi)
        else:
            effpot[i] = Veff(x[i], xi)
    return effpot

# Define the effective force functions
#Force $F = \frac{8AB\mathrm{e}^{2A \left(x - {\xi}\right)} \left(\left(2Ax - 2A{\xi} - 3\right) \mathrm{e}^{4A \left(x - {\xi}\right)} + \left(8Ax - 8A{\xi}\right) \mathrm{e}^{2A \left(x - {\xi}\right)} + 2Ax - 2A{\xi} + 3\right)}{\left(\mathrm{e}^{2A \left(x - {\xi}\right)} - 1\right)^{4}}$
def Feff(x, xi=0):
    temp = x - xi
    return (8 * A * B * np.exp(2 * A * temp) * ((2 * A * temp - 3) * np.exp(4 * A * temp) + (8 * A * temp) * np.exp(2 * A * temp) + 2 * A * temp + 3) /
            (np.exp(2 * A * temp) - 1) ** 4)

#Force $F = \frac{8A^{2} B \left(x - {\xi}\right)}{15}$
def Feff_tay(x, xi=0):
    return (8 * A ** 2 * B * (x - xi)) / 15

# Define the effective force function that switches between Feff and Feff_tay
def Force_eff_no_trap(x, L, R, xi=0):
    eff_force = np.zeros_like(x)
    for i in range(len(x)):
        if L <= x[i] <= R:
            eff_force[i] = Feff_tay(x[i], xi)
        else:
            eff_force[i] = Feff(x[i], xi)
    return eff_force

window=0.5
#plt.plot(x:=np.linspace(0, window, 1000), suspicious_bs(np.linspace(0, window, 1000)))
#plt.plot(x, numerator_suspicios_bs(x))
#plt.ylim(-1,2)
#plt.show()
#plt.close()

def sech_fit(x, A, b):
    return A * sech(b * x)

def sech2_fit(x, A, b):
    return A * sech(b * x) ** 2

sech = lambda x: 1/np.cosh(x)

# Generate data
x = np.linspace(-10, 10, 1000).reshape(-1, 1)
y = Veff_no_trap(x, L_tay, R_tay)

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
print(f"A_sech2 = {A_sech2}, b_sech2 = {b_sech2}")

#model_func = lambda x: np.sin(np.sin((np.sin(sech(0.45118 * x)) - 0.02943) * sech(x)))
#model_func = lambda x, A, b: np.sin(np.sin((np.sin(sech(A * x)) - B) * sech(x)))
#model.predict = model_func

# Fit PySR function
#params_PySR, _ = curve_fit(model.predict, x.ravel(), y.ravel(), p0=[1, 1])
#A_PySR, b_PySR = params_PySR

# Generate fitted curves
y_fit_sech = sech_fit(x, A_sech, b_sech)
y_fit_sech2 = sech2_fit(x, A_sech2, b_sech2)
#y_fit_PySR = model.predict(x, A_PySR, b_PySR)

# Plot results
plt.figure(figsize=(8, 5))
mt_part = .5*Omega*Omega*x*x
plt.plot(x, y+mt_part, label="Original", color="black")

# Compute MSE
#mse_pysr = np.mean((y - y_fit_PySR)**2)
mse_sech = np.mean((y - y_fit_sech)**2)
mse_sech2 = np.mean((y - y_fit_sech2)**2)

#plt.plot(x, y_fit_PySR+mt_part, "--", label=r"$V(x) = \mathrm{sin}(\mathrm{sin}((\mathrm{sin}(\mathrm{sech}("f"{A_PySR}"r" \cdot x)) - "f"{b_PySR}"r") \cdot \mathrm{sech}(x)))$"f', MSE = {mse_pysr:.3e}', color="red")
plt.plot(x, y_fit_sech+mt_part, label=r'$V(x) = A\cdot\mathrm{sech}(bx), $'f'$A={A_sech:.3f}$, $b={b_sech:.3f}$'f', MSE = {mse_sech:.3e}', linestyle='dashed')
plt.plot(x, y_fit_sech2+mt_part, label=r'$V(x) = A\cdot\mathrm{sech}^2(bx), $'f'$A={A_sech2:.3f}, b={b_sech2:.3f}$'f', MSE = {mse_sech2:.3e}', linestyle='dotted')

print(min(y_fit_sech2+mt_part))
print(min(y+mt_part))
plt.legend()
plt.xlabel("x")
plt.ylabel("V(x)")
#$\mathrm{sin}(\mathrm{sin}((\mathrm{sin}(\mathrm{sech}(0.45118 \cdot x)) - 0.02943) \cdot \mathrm{sech}(x)))$
plt.tight_layout()
plt.savefig("PaulVariationalPotentialPlot.png",dpi=5*96)
system("open PaulVariationalPotentialPlot.png")

