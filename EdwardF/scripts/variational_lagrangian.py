import sympy as sp
from sympy import sech, tanh, sinh, cosh, simplify, Rational

def declutter(latex_string):
    """Customizes LaTeX output by replacing specific patterns and adjusting parentheses."""
    declutter_map = {
        r"\frac{d}{d t} \mathcal{A}": r"\dot{\mathcal{A}}",
        r"\frac{d}{d t} \mathcal{X}": r"\dot{\mathcal{X}}",
        r"\frac{d}{d t} \mathcal{B}": r"\dot{\mathcal{B}}",
        r"\frac{d}{d t} \mathcal{C}": r"\dot{\mathcal{C}}",
        r"\frac{d^{2}}{d t^{2}} \mathcal{A}": r"\ddot{\mathcal{A}}",
        r"\frac{d^{2}}{d t^{2}} \mathcal{X}": r"\ddot{\mathcal{X}}",
        r"\frac{d^{2}}{d t^{2}} \mathcal{B}": r"\ddot{\mathcal{B}}",
        r"\frac{d^{2}}{d t^{2}} \mathcal{C}": r"\ddot{\mathcal{C}}",
#        r"A": r"\mathcal{A}",
        r"A_{0}": r"A",
#        r"X": r"\mathcal{X}",
#        r"B": r"\mathcal{B}",
#        r"C": r"\mathcal{C}",
        r"{\left(t \right)}": r"",
        r"1.0 x": r"x",  # Remove floating-point 1.0
        r"1.0 V": r"V",  # Remove floating-point 1.0
        r"1.0 \mathcal{A}": r"\mathcal{A}",  # Remove floating-point 1.0
        r"1.0 \mathcal{X}": r"\mathcal{X}",  # Remove floating-point 1.0
        r"1.0 \mathcal{B}": r"\mathcal{B}",  # Remove floating-point 1.0
        r"1.0 \mathcal{C}": r"\mathcal{C}",  # Remove floating-point 1.0
        r"\mathcal{A}(t)": r"\mathcal{A}",  # Remove (t)
        r"\mathcal{X}(t)": r"\mathcal{X}",  # Remove (t)
        r"\mathcal{B}(t)": r"\mathcal{B}",  # Remove (t)
        r"\mathcal{C}(t)": r"\mathcal{C}",  # Remove (t)
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
L_dens_symbol, L_a_symbol, u, x, t, meff = sp.symbols('\\mathcal{L} L_a u x t m_{\\text{eff}}', real=True)
A, X, B, C = sp.symbols('\\mathcal{A} \\mathcal{X} \\mathcal{B} \\mathcal{C}', real=True, cls=sp.Function)  # Variational parameters as functions of t
#V = sp.Function('V')(x)  # External arbitrary potential V(x)
Omega, A0, b = sp.symbols('Omega A b', real=True)
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

#width-matching assumption  b ≔ 𝒜(t)
L_dens = L_dens.subs(b, A(t))
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

def I_mixed_shift(A_t, X_t):
    """
    ∫_{-∞}^{∞} sech⁴(A_x * X_t) · sech²(u + A_t*X_t) du
    """
    s = A_t * X_t
    num = 32*sp.exp(2*s)*(sp.exp(6*s)
          + (9 - 12*s)*sp.exp(4*s)
          + (-12*s - 9)*sp.exp(2*s) - 1)
    den = 3*A_t*(sp.exp(10*s) - 5*sp.exp(8*s) + 10*sp.exp(6*s)
          - 10*sp.exp(4*s) + 5*sp.exp(2*s) - 1)
    return num / den

L_terms = sp.Add.make_args(L_dens)   # split the density into its Σ pieces
wild_q = sp.Wild('q', properties=[lambda k: k.is_integer])

total = 0

#Integration loop:
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

    integrand = sp.cancel(integrand.subs(u, A(t) * (x - X(t))))
    # And since du = A(t) dx, we multiply by A(t)
    coeff *= A(t)
    
    # (5) sech⁴(A·x)·sech²(A·x - A·X)
    test = (sech(A(t)*x)**2) * (sech(A(t)*(x-X(t)))**4)
    print(f"test = {test}")
    if sp.cancel(integrand - test) == 0:
        print(f"COEFF = {coeff}, INTEGRAND = {integrand}")
        total += coeff * I_mixed_shift(A(t), X(t))
        continue
    else:
        print(f"failed, COEFF = {coeff}, INTEGRAND = {integrand}")
        
    # (6) fallback – let SymPy try the definite integral itself
    integrand = sp.Integral(integrand, (x, -sp.oo, sp.oo))

    # if val is still an unevaluated Integral, SymPy couldn’t do it;
    # we keep the symbolic object so the algebra downstream still works
    total += coeff * integrand # works for both numbers and Integral(...)

L_a = total
print("Effective Lagrangian:", declutter(sp.multiline_latex((L_a_symbol), L_a, 2)))

coords = (A(t), X(t), B(t), C(t))
EL_eqns = []

for q in coords:
    qdot = sp.diff(q, t)
    EL = sp.diff(sp.diff(L_a, qdot), t) - sp.diff(L_a, q)
    EL_eqns.append(sp.cancel(sp.expand(EL)))

# pretty print --------------------------------------------------
for q, eq in zip(coords, EL_eqns):
    print(f"{'🐨'*10}\nq={q}\n{'🐨'*10}")
    print(sp.multiline_latex(0, eq, 2))

# -------------------------------------------------------------------------
# constants and substitutions dictated by the EL constraints observed above
# -------------------------------------------------------------------------
A0 = sp.symbols(r'\mathcal{A}_0', positive=True)   # constant width

freeze_A = {A(t): A0, sp.diff(A(t), t): 0}         # only A is frozen here
full_subs = {**freeze_A,
             C(t):  sp.diff(X(t), t),              # C → Ẋ  (after EL’s)
             sp.diff(C(t), t): sp.diff(X(t), t, 2)}

# ---------- 1.  derive EL‑eqn for X BEFORE imposing C = Ẋ ----------
L_c = L_a.subs(freeze_A)                           # A frozen, C kept
Xdot = sp.diff(X(t), t)
EL_X_raw = sp.simplify(
             sp.diff(sp.diff(L_c, Xdot), t) - sp.diff(L_c, X(t)))

# now impose the kinematic relation C = Ẋ
EL_X = sp.simplify(EL_X_raw.subs({C(t): Xdot,
                                  sp.diff(C(t), t): sp.diff(X(t), t, 2)}))

# ---------- 2.  solve for  Ẍ  --------------------------------------
Xddot = sp.diff(X(t), t, 2)
sol_X = sp.solve(EL_X, Xddot)
if not sol_X:
    raise RuntimeError("❌  SymPy could not isolate Xddot. Check algebra.")
Xddot_expr = sp.cancel(sp.expand(sol_X[0]))

print("\n🎉  Newton–type equation obtained:")
print(declutter(sp.multiline_latex(Xddot, Xddot_expr, 1)))

# ---------- 3.  substitute ALL static relations in L --------------
L_eff = L_a.subs(full_subs)
print("\nEffective Lagrangian with all constraints:")
print(declutter(sp.multiline_latex(L_a_symbol, L_eff, 2)))

# ---------- 4.  obtain 𝔅̇ from the A‑equation -----------------------
Adot = sp.diff(A(t), t)            # still appears in EL_A before freeze
EL_A = sp.cancel(
          sp.expand(sp.diff(sp.diff(L_eff, Adot), t) - sp.diff(L_eff, A0)))

Bdot = sp.diff(B(t), t)
sol_B = sp.solve(EL_A, Bdot)
if not sol_B:
    raise RuntimeError("❌  Could not solve for 𝔅̇.")
Bdot_expr = sp.cancel(sp.expand(sol_B[0]))

# optional pretty partial‑fraction trick
s, z = sp.symbols('s z')
Bdot_expr = (Bdot_expr
             .subs(A0*X(t), s)
             .subs(sp.exp(2*s), z)
             .apart(z)
             .subs({z: sp.exp(2*A0*X(t)), s: A0*X(t)}))

print("\nExpression for $\\dot{\\mathcal{B}}(t)$:")
print(declutter(sp.multiline_latex(Bdot, Bdot_expr, 2)))

# ---------- 5.  pretty partial‑fraction form for  Ẍ  -----------------
s, z = sp.symbols('s z')

Xddot_pf = (Xddot_expr
            .subs(A0*X(t), s)       # s := 𝒜₀ X
            .subs(sp.exp(2*s), z)   # z := e^{2s}
            .apart(z)               # partial‑fraction in z
            .subs({z: sp.exp(2*A0*X(t)),
                   s: A0*X(t)}))    # restore original variables

print("\nPartial‑fraction form of $\\ddot{\\mathcal{X}}(t)$:")
print(declutter(sp.multiline_latex(Xddot, Xddot_pf, 3)))
assert(Xddot_expr.equals(Xddot_pf))

# ---------- 6.  extract the effective potential directly -------------
m_eff = sp.Rational(4,3)*A0        # from  T = (2/3)A0 Ẋ²
U_from_int = sp.integrate(-m_eff*Xddot_pf, (X(t),))
print("\nU_eff(𝓧) (from integration):")
U_from_int = (U_from_int.subs(A0*X(t), s)
                .subs(sp.exp(2*s), z)
                .apart(z)
                .subs({z: sp.exp(2*A0*X(t)), s: A0*X(t)}))
print(declutter(sp.multiline_latex(sp.Symbol(r'U_{\text{eff}}'), U_from_int, 3)))


# set EVERY explicit time-derivative to zero
zeros = {sp.diff(X(t), t): 0,
         sp.diff(X(t), t, 2): 0,
         sp.diff(B(t), t): 0}

U_eff = sp.simplify(-L_eff.subs(zeros))   # flip sign:  L = T - U
U_eff = sp.cancel(U_eff)

# compact partial-fraction version for neat LaTeX
s, z = sp.symbols('s z')
U_pf = (U_eff
        .subs(A0*X(t), s)
        .subs(sp.exp(2*s), z)
        .apart(z)
        .subs({z: sp.exp(2*A0*X(t)), s: A0*X(t)}))

print("\nRaw U_eff(𝓧):")
print(declutter(sp.multiline_latex(sp.Symbol(r'U_{\text{eff}}'), U_eff, 3)))

print("\nPretty U_eff(𝓧):")
print(declutter(sp.multiline_latex(sp.Symbol(r'U_{\text{eff}}'), U_eff, 3)))

# ---------- sanity:  m_eff * Ẍ  + ∂U/∂X  = 0  ----------------------
chk = sp.simplify(m_eff*Xddot_pf + sp.diff(U_eff, X(t)))
assert chk == 0
print("\n✅  m_eff Ẍ + ∂U/∂X = 0   (check passed)")

# compare directive integration potential with zero'd Lagrangian potential
diff = sp.simplify(U_from_int - U_eff)
print("\nDifference between the two U’s:", sp.latex(diff.expand()))

# -----------------------------------------------------------------
#  Clean 2-nd-order Taylor expansion  U_eff(ξ + s)  at  s = 0
# -----------------------------------------------------------------
Xs  = sp.symbols('Xs')                         # dummy series variable
U_T2 = sp.series(U_from_int.subs({X(t): Xs}), Xs, 0, 3).removeO().expand().cancel()
U_T2 = U_T2.subs(Xs, X(t))
print(declutter(sp.multiline_latex(sp.Symbol(r'\text{Taylor Expansion: }'), U_T2, 3)))
print(U_T2)
#How do I do this line up above on my `U_eff` to get the same result `2*A0*(56*A + X**2*(-32*A*A0**2 + 35*Omega**2))/105`?
