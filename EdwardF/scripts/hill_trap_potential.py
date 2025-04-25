import numpy as np
import matplotlib.pyplot as plt
import os

def hill_trap_potential(x, A, b, Omega):
    return 0.5 * Omega * Omega * x * x + A * sech(b * x) ** 2

sech = lambda x: 1/np.cosh(x)

# Generate data
x = np.linspace(-10, 10, 1000)
plt.plot(x, hill_trap_potential(x, A = 1, b = 1, Omega = 0.2), label = r"$A = 1, b = 1, \Omega = 0.2$")
plt.title(r"$V(x) = \frac{1}{2}\Omega^2 x^2 + A\cdot \mathrm{sech}^2(bx)$")
plt.xlabel("$x$")
plt.ylabel(r"$V(x)$")
plt.legend()
plt.savefig("hill_trap_potential.svg")
os.system(f"rsvg-convert -f pdf -o hill_trap_potential.pdf hill_trap_potential.svg")
os.system("rm hill_trap_potential.svg")
os.system("open hill_trap_potential.pdf")
