# ------------------------------------------------------------------
#  Visual check:  U_eff(X)  vs.  external trap V(x)
#  parameters taken directly from the manuscript
# ------------------------------------------------------------------
import numpy as np, matplotlib.pyplot as plt, sympy as sp
from os import system as sys


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
xgrid  = np.linspace(-4, 4, 801)
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

sys("open U_eff_vs_V_ext.pdf")
