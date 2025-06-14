import sympy as sp

def declutter(latex_string):
    """Customizes LaTeX output by replacing specific patterns and adjusting parentheses."""
    declutter_map = {
        r"\frac{d}{d t} A": r"\dot{\mathcal{A}}",
        r"\frac{d}{d t} X": r"\dot{\mathcal{X}}",
        r"\frac{d}{d t} B": r"\dot{\mathcal{B}}",
        r"\frac{d}{d t} C": r"\dot{\mathcal{C}}",
        r"A": r"\mathcal{A}",
        r"X": r"\mathcal{X}",
        r"B": r"\mathcal{B}",
        r"C": r"\mathcal{C}",
        r"{\left(t \right)}": r"(t)",
        r"1.0 x": r"x",  # Remove floating-point 1.0
        r"1.0 V": r"V",  # Remove floating-point 1.0
        r"1.0 \mathcal{A}": r"\mathcal{A}",  # Remove floating-point 1.0
        r"1.0 \mathcal{X}": r"\mathcal{X}",  # Remove floating-point 1.0
        r"1.0 \mathcal{B}": r"\mathcal{B}",  # Remove floating-point 1.0
        r"1.0 \mathcal{C}": r"\mathcal{C}",  # Remove floating-point 1.0
        r"1.0 \dot": r"\dot",  # Remove floating-point 1.0
        r"2.0": r"2",  # Ensure integer coefficients
        r"0.5": r"\frac{1}{2}"  # Convert 0.5 to fraction
    }
    for str_to_replace, replacement in declutter_map.items():
        latex_string = latex_string.replace(str_to_replace, replacement)
    
    return latex_string


# Define symbols
L_dens, x, t = sp.symbols('\\mathcal{L} x t', real=True)
A, X, B, C = sp.symbols('A X B C', real=True, cls=sp.Function)  # Variational parameters as functions of t
V = sp.Function('V')(x)  # External potential V(x)

# Define the ansatz
chi = A(t) * (x - X(t))  # Argument of sech^2
u_a = A(t) * sp.sech(chi)**2 * sp.exp(sp.I * (B(t) + C(t) * x))
u_a_conj = A(t) * sp.sech(chi)**2 * sp.exp(-sp.I * (B(t) + C(t) * x))

# Compute derivatives
u_t = sp.diff(u_a, t)
u_x = sp.diff(u_a, x)
u_t_conj = sp.diff(u_a_conj, t)
u_x_conj = sp.diff(u_a_conj, x)

# Lagrangian density terms
time_term = (sp.I / 2) * (u_a_conj * u_t - u_a * u_t_conj)
spatial_term = -(1 / 2) * u_x_conj * u_x
nonlinear_term = (1 / 2) * (u_a_conj**2) * (u_a**2)
potential_term = -V * u_a_conj * u_a

# Total Lagrangian density
L = time_term + spatial_term + nonlinear_term + potential_term

# Simplify before substitution
L_before = sp.cancel(sp.trigsimp(sp.cancel(L)).rewrite(sp.cosh, sp.sech))
L = L_before.subs(A(t) * (x - X(t)), chi)
L_dens_latex = declutter(sp.multiline_latex(L_dens, L, 1))
print("L density LaTeX:", L_dens_latex)

# Integrate over x from -infty to infty
# Use substitution: chi = A(t) * (x - X(t)), dx = dchi / A(t)
#L_eff = sp.integrate(L / A(t), (chi, -sp.oo, sp.oo))
#
## Simplify the effective Lagrangian
#L_eff = sp.simplify(L_eff)
#
## Convert to LaTeX
#latex_code = sp.latex(L_eff, mode='inline')

# Print the LaTeX code
#print(latex_code)

