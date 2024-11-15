import sympy as sp
from sympy import *

# Define symbols
t = sp.symbols('t')

# Define the expression
expr = tanh(((1.761061 + (t * cos(4.590427))) * (t / (t + sech(10.000000)))))


# Simplify the expression
simplified_expr = sp.simplify(expr)

# Convert to LaTeX
latex_expr = sp.latex(simplified_expr)
print(latex_expr)

