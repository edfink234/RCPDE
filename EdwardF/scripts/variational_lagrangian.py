import sympy as sp
from sympy import sech, tanh, sinh, cosh, simplify, Rational

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
        r"1.0 u": r"u",   # Remove floating-point 1.0
        r"2.0": r"2",  # Ensure integer coefficients
        r"0.5": r"\dfrac{1}{2}",  # Convert 0.5 to fraction,
        r"\\": r"\nonumber \\", # Removes the automatic numbering from the align environment
        r"\frac": r"\dfrac"
    }
    for str_to_replace, replacement in declutter_map.items():
        latex_string = latex_string.replace(str_to_replace, replacement)
    
    return latex_string

# Define symbols
L_dens_symbol, L_a_symbol, u, x, t = sp.symbols('\\mathcal{L} L_a u x t', real=True)
A, X, B, C = sp.symbols('A X B C', real=True, cls=sp.Function)  # Variational parameters as functions of t
#V = sp.Function('V')(x)  # External arbitrary potential V(x)
Omega, A0, b = sp.symbols('Omega A0 b', real=True)
V = Rational(1, 2) * Omega**2 * x**2 + A0 * sech(b * x)**2

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
spatial_term = Rational(-1, 2) * u_x_conj * u_x
nonlinear_term = Rational(1, 2) * (u_a_conj**2) * (u_a**2)
potential_term = -V * u_a_conj * u_a

# Total Lagrangian density
L_dens = time_term + spatial_term + nonlinear_term + potential_term
L_dens = sp.cancel(sp.expand(L_dens)).rewrite(sp.cosh, sp.sech)
L_dens_latex = declutter(sp.multiline_latex(L_dens_symbol, L_dens, 1)) #https://github.com/sympy/sympy/blob/master/sympy/printing/latex.py#L3192
print("L density LaTeX:", L_dens_latex)

# u = A(t) * (x - X(t))  =>  x = u / A(t) + X(t) => dx = du / A(t)
subs_dict = {x: u / A(t) + X(t)}
L_dens = L_dens.subs(subs_dict) * (1 / A(t))
L_dens_latex = declutter(sp.multiline_latex(L_dens_symbol, L_dens, 1))
print("L density LaTeX after u-sub:", L_dens_latex)
L_dens = sp.expand(sp.cancel(L_dens))
L_dens_latex = declutter(sp.multiline_latex(L_dens_symbol, L_dens, 1))
print("L density LaTeX after u-sub and simplification:", L_dens_latex)

# --- exact (–∞,∞) integrals of sech^n --------------------------
def I_sech_even(power):
    """
    ∫_{-∞}^{∞} sech^power(u) du  (power must be even).
    Uses z = tanh u  ⇒  sech^2 u du = dz.
    """
    assert power % 2 == 0, "power must be even"
    m = power // 2
    z = sp.symbols('z')
    return sp.integrate((1 - z**2)**(m - 1), (z, -1, 1))

def I_u2_sech4():
    """
    ∫_{-∞}^{∞} u^2 sech^4 du 
    """
    return (sp.pi**2 - 6)/9        # exact

L_terms = sp.Add.make_args(L_dens)   # split the density into its Σ pieces
wild_q = sp.Wild('q', properties=[lambda k: k.is_integer])

total = 0
for i, term in enumerate(L_terms):
    coeff, integrand = term.as_independent(u)

    # (1) odd integrand → integral = 0
    if integrand.subs(u, -u) == -integrand:
        continue

    # (2) pure sech^n
    if integrand.is_Pow and integrand.base == sech(u):
        n = int(integrand.exp)
        total += coeff * I_sech_even(n)
        continue

    # (3)  tanh²·sech^q
    hit = integrand.match(tanh(u)**2 * sech(u)**wild_q)
    if hit:
        q = int(hit[wild_q])
        total += coeff * (I_sech_even(q) - I_sech_even(q + 2))
        continue
    

    # (4)  u^2 sech^4(u)
    if integrand == u**2 * sech(u)**4:
        print(f"COEFF = {coeff}, INTEGRAND = {integrand}")
        total += coeff * I_u2_sech4()
        continue

    # (5) fallback – let SymPy try the definite integral itself
    integrand_x = sp.cancel(integrand.subs(u, A(t) * (x - X(t))))
    # And since du = A(t) dx, we multiply by A(t)
    val = A(t) * sp.Integral(integrand_x, (x, -sp.oo, sp.oo))

    # if val is still an unevaluated Integral, SymPy couldn’t do it;
    # we keep the symbolic object so the algebra downstream still works
    total += coeff * val # works for both numbers and Integral(...)

L_a = total
print("Effective Lagrangian:", declutter(sp.multiline_latex(sp.simplify(L_a_symbol), L_a, 2)))



