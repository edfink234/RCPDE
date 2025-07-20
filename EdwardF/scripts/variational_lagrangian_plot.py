# ------------------------------------------------------------------
#  Visual check:  U_eff(X)  vs.  external trap V(x)
#  parameters taken directly from the manuscript
# ------------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, sympy as sp
from os import system as sys
from scipy.optimize import curve_fit

def sci_to_latex(sci_str):
    try:
        coeff, exp = sci_str.split('e')
        latex_str = rf"${coeff} \times 10^{{{int(exp)}}}$"
        return latex_str
    except ValueError:
        return "nan"

# parameters
Ω, A0, A_val = 0.2, 1.0, 1.0      # b = A0 = 1   and  A = 1
X = sp.symbols('X')

# analytic U_eff(X) from Eq. (6.xxx)  (already derived)
U_eff_sym = (
    -256*A_val*A0**2*X/(sp.exp(2*A0*X)-1)**5
    -256*A_val*A0*(2*A0*X-1)/(sp.exp(2*A0*X)-1)**3
    -128*A_val*A0*(5*A0*X-1)/(sp.exp(2*A0*X)-1)**4
    -32*A_val*A0*(12*A0*X-13)/(3*(sp.exp(2*A0*X)-1)**2)
    +32*A_val*A0/(3*(sp.exp(2*A0*X)-1))
    +2*Ω**2*A0*X**2/3
)
U_eff = sp.lambdify(X, U_eff_sym, 'numpy')

# external trap
def V_ext(x, Ω_now):
    return 0.5*Ω_now**2 * x**2 + A_val/np.cosh(A0*x)**2

# grid and curves
xgrid  = np.linspace(-10, 10, 1001)
U_full = U_eff(xgrid)                       # Ω = 0.2
V_full = V_ext(xgrid, Ω)                    # Ω = 0.2
U_def  = U_eff(xgrid) - (2*Ω**2*A0*xgrid**2/3)  # set Ω→0 in U
V_def  = V_ext(xgrid, 0.0)                  # Ω = 0

# plotting
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)

# (a) full trap
ax[0].plot(xgrid, U_full, label=r'$U_{\mathrm{eff}}(\mathcal{X})$', lw=2)
ax[0].plot(xgrid, V_full, '--', label=r'$V_{\mathrm{ext}}(x)$', lw=2)
ax[0].set_title(r'Hill Component + $\Omega^2$ Term')
ax[0].legend()

# (b) defect only
ax[1].plot(xgrid, U_def, label=r'$U_{\mathrm{eff}}(\mathcal{X}) - \dfrac{2\Omega^2\mathcal{A}_0\mathcal{X}^2}{3}$', lw=2)
ax[1].plot(xgrid, V_def, '--', label=r'$V_{\mathrm{ext}}(x) - \dfrac{\Omega^2 x^2}{2}$', lw=2)
ax[1].set_title('Hill Component Only')
ax[1].legend()

for a in ax:
    a.set_ylabel(r'Potential energy')
ax[1].set_xlabel(r'$x$ or $\mathcal{X}$')
fig.suptitle(r'$V(x)\,$ vs $\,U_{\mathrm{eff}}(\mathcal{X})$ for $\Omega = 0.2$, $A = b = \mathcal{A}_0 = 1$')
plt.tight_layout()
plt.savefig("U_eff_vs_V_ext.pdf")
plt.close()
sys("open U_eff_vs_V_ext.pdf")

# ---------------- U_eff(𝓧) near 𝓧 = ξ ----------------
# plotting
fig, ax = plt.subplots(2, 1, figsize=(6, 8), sharex=True)
# Define tight grid near X = 0
x_zoom = np.linspace(-10, 10, 20000)
U_zoom = U_eff(x_zoom)

ax[0].plot(x_zoom, U_zoom, lw=2, color='darkred')
ax[0].set_ylabel(r'$U_{\mathrm{eff}}(\mathcal{X})$')
ax[0].set_title(r'$U_{\mathrm{eff}}(\mathcal{X})$')
ax[0].grid(True)

x_zoom_squared = x_zoom*x_zoom
U_taylor = (
    - (64 * A_val*A0*A0*A0 * x_zoom_squared) / 105
    + ((16 * A_val*A0) / 15)
    + ((2 * Ω*Ω*A0 * x_zoom_squared) / 3)
)
rho = 2e-1
use_taylor = (np.abs(x_zoom) < rho)
U_patch = np.where(use_taylor, U_taylor, U_eff(x_zoom))
ax[1].plot(x_zoom, U_patch, lw=2, color='darkgreen')
ax[1].set_ylabel(r'$U_{\mathrm{patched}}(\mathcal{X})$')
ax[1].set_title(r'$U_{\mathrm{patched}}(\mathcal{X})$ with $\rho = $'+sci_to_latex(f"{rho:.2e}"))
ax[1].set_xlabel(r'$\mathcal{X}$')
ax[1].grid(True)
plt.suptitle(r'$U_{\mathrm{eff}}(\mathcal{X})$ vs $U_{\mathrm{patched}}(\mathcal{X})$''\n'r'$\left(N=2\times 10^4 \: \mathrm{points}, \: U_{\mathrm{Taylor}}(\mathcal{X}) \: \mathrm{at} \: |\mathcal{X}| < \rho\right)$', fontsize=12)

plt.tight_layout()
plt.savefig("U_eff_unpatched_vs_patched.pdf")
plt.close()
sys("open U_eff_unpatched_vs_patched.pdf")

# ---------------- Fixed variational potential parameters ----------------
Ω_var = 0.2   # omega in U_eff
A0 = 1.0      # A_0 in U_eff
A_val = 1.0   # A in U_eff

# Define U_eff(X) with fixed parameters
def U_eff_fixed(x):
    exp_term = np.exp(2 * A0 * x) - 1
    return (
        -256 * A_val * A0**2 * x / exp_term**5
        -256 * A_val * A0 * (2 * A0 * x - 1) / exp_term**3
        -128 * A_val * A0 * (5 * A0 * x - 1) / exp_term**4
        -32 * A_val * A0 * (12 * A0 * x - 13) / (3 * exp_term**2)
        + 32 * A_val * A0 / (3 * exp_term)
        + 2 * Ω_var**2 * A0 * x**2 / 3
    )

# Define external potential V_ext(x; A, b, Ω) for fitting
def V_ext_fit(x, A, b, Ω):
    return 0.5 * Ω**2 * x**2 + A / np.cosh(b * x) ** 2

# Set x-grid and filter out regions with singularities in U_eff
xgrid = np.linspace(-10, 10, 1001)
U_ref = U_eff_fixed(xgrid)
valid_mask = np.isfinite(U_ref)
x_fit = xgrid[valid_mask]
x_zoom_squared = x_fit*x_fit
U_taylor = (
    - (64 * A_val*A0*A0*A0 * x_zoom_squared) / 105
    + ((16 * A_val*A0) / 15)
    + ((2 * Ω*Ω*A0 * x_zoom_squared) / 3)
)
rho = 2e-1
use_taylor = (np.abs(x_fit) < rho)
U_fit = np.where(use_taylor, U_taylor, U_ref[valid_mask])

# Fit external potential parameters (A, b, Ω) to U_eff
popt, _ = curve_fit(V_ext_fit, x_fit, U_fit, p0=[1.0, 1.0, 0.2])
A_fit, b_fit, Ω_fit = popt
V_best = V_ext_fit(xgrid, A_fit, b_fit, Ω_fit)

# Output fitted parameters
print(A_fit, b_fit, Ω_fit)
mse = np.mean((V_ext_fit(x_fit, *popt) - U_fit) ** 2)
print(f"Mean squared error of fit: {mse}")

# ------------------------- Plotting ------------------------------------
fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(xgrid, U_ref, label=r'$U_{\mathrm{patched}}(\mathcal{X})$', lw=2)
ax3.plot(xgrid, V_best, '--', label=fr'$V(x)$ ($A={A_fit:.2f}$, $b={b_fit:.2f}$, $\Omega={Ω_fit:.2f}$)', lw=2)
ax3.set_xlabel(r'$x$ or $\mathcal{X}$')
ax3.set_ylabel('Potential energy')
ax3.set_title(r'Fit of $V$ to $U_{\mathrm{patched}}$ (fixed to $\mathcal{A}_0=1$, $A=1$, $\Omega=0.2$)'f",\nMSE = {sci_to_latex(f'{mse:.2e}')}")
ax3.legend()
plt.tight_layout()
plt.savefig("V_ext_fit_vs_U_eff.pdf")
plt.close()
sys("open V_ext_fit_vs_U_eff.pdf")


