from os import system
import numpy as np
from scipy.optimize import fsolve
#for x_0 in [1.5, 1.0, -0.5]:#np.linspace(-1.0, 1.5, 6):
#    with open("temp.txt", "w") as f:
#        f.write(f"{x_0}\n")
#    system("python LearnInvertedPenulum.py")

Omega = 0.2
def func(x):
    return Omega*Omega*x - 2 * np.cosh(x)**(-2) * np.tanh(x)

# Initial guess for the root (you might need to adjust this)
x0 = 1.0

# Find the root
root, info, ier, mesg = fsolve(func, x0, full_output=True)

if ier == 1:
    print(f"Root found: {root[0]}")
    with open("temp.txt", "w") as f:
        f.write(f"{root[0]}\n")
    system("python LearnInvertedPenulum.py")
else:
    print(f"Root finding failed: {mesg}")


