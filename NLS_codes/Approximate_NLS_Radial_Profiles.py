import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os

# Define the symbol
x = sp.symbols('x')

# Define the expressions
expr1 = sp.acos(sp.cos(sp.tanh(x))**(sp.sech(sp.sech(x))))
expr2 = sp.sin(sp.acos(sp.sqrt(sp.sech(x))))
expr3 = sp.cos(sp.asin(sp.sqrt(sp.sech(x))))
expr4 = sp.tanh(x)**2
expr5 = (sp.sqrt(sp.sech(x)) - sp.sech(x))/x
expr6 = sp.sqrt(sp.sqrt(sp.tanh(x))) - sp.cos(sp.sech(x))
expr7 = sp.sech((sp.log(sp.exp(x))/(sp.sech(x) - sp.cos(x))))
expr8 = sp.sin(sp.acos(sp.tanh(x)/x))

# Convert to lambdified functions for numerical evaluation
f1 = sp.lambdify(x, expr1, "numpy")
f2 = sp.lambdify(x, expr2, "numpy")
f3 = sp.lambdify(x, expr3, "numpy")
f4 = sp.lambdify(x, expr4, "numpy")
f5 = sp.lambdify(x, expr5, "numpy")
f6 = sp.lambdify(x, expr6, "numpy")
f7 = sp.lambdify(x, expr7, "numpy")
f8 = sp.lambdify(x, expr8, "numpy")

# Define the range for x
x_vals = np.linspace(0, 15, 400)  # Avoid division by zero for expr5 and expr8

# Evaluate the expressions
y1 = f1(x_vals)
y2 = f2(x_vals)
y3 = f3(x_vals)
y4 = f4(x_vals)
y5 = f5(x_vals)
y6 = f6(x_vals)
y7 = f7(x_vals)
y8 = f8(x_vals)

# Plot the expressions

plt.plot(x_vals, y1, label=r'$ \arccos\left(\cos\left(\tanh\left(r\right)\right)^{\left(\operatorname{sech}\left(\operatorname{sech}\left(r\right)\right)\right)}\right) $')
plt.plot(x_vals, y2, label=r'$ \sin\left(\arccos\left(\sqrt{\operatorname{sech}\left(r\right)}\right)\right) $')
plt.plot(x_vals, y3, label=r'$ \cos\left(\arcsin\left(\sqrt{\operatorname{sech}\left(r\right)}\right)\right) $')
plt.plot(x_vals, y4, label=r'$ \tanh^{2}\left(r\right) $', linestyle = ":")
plt.plot(x_vals, y5, label=r'$ \frac{\sqrt{\operatorname{sech}\left(r\right)}-\operatorname{sech}\left(r\right)}{r} $')
plt.plot(x_vals, y6, label=r'$ \sqrt{\sqrt{\tanh\left(r\right)}}-\cos\left(\operatorname{sech}\left(r\right)\right) $')
plt.plot(x_vals, y7, label=r'$ \operatorname{sech}((\log(\exp(r))/(\operatorname{sech}(r)-\cos(r)))) $')
plt.plot(x_vals, y8, label=r'$ \sin(\arccos((\tanh(r)/r))) $')

plt.xlim(0, 15)
plt.ylim(0, 1)

plt.xlabel('r')
plt.ylabel('R(r)')
plt.legend(loc='best')
plt.title('Approximate NLS Radial Profiles')
plt.grid(True)
plt.savefig("Approximate_NLS_Radial_Profiles.svg")
os.system("rsvg-convert -f pdf -o Approximate_NLS_Radial_Profiles.pdf Approximate_NLS_Radial_Profiles.svg")
os.system("rm Approximate_NLS_Radial_Profiles.svg")
os.system("open Approximate_NLS_Radial_Profiles.pdf")
