import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def rate(t, u, params):
    u = u.reshape(-1, 1)
#    print(f"u.shape = {u.shape}")
    nx = len(u) // 2
    Dxx = params['Dxx']
    Vpot = params['Vpot']
    psi = u[:nx] + 1j * u[nx:2*nx]
#    print(f"Dxx.shape = {Dxx.shape}, psi.shape = {psi.shape}, Vpot.shape = {Vpot.shape}")
    out = -1j * (-0.5 * Dxx @ psi - np.abs(psi)**2 * psi + Vpot * psi)
    return np.vstack([np.real(out), np.imag(out)]).flatten()

# Clear workspace and set up parameters
x = np.linspace(-20, 20, 1201).reshape(-1, 1)
nx = len(x)
dx = x[1] - x[0]

one = np.ones((nx, 1))
Dxx = sp.diags([one.flatten(), -2*one.flatten(), one.flatten()], [-1, 0, 1], shape=(nx, nx)) / dx**2

a = 1
omg = 0.2
x0 = 0
Vpot = 0.5 * omg**2 * x**2 + a * (1 / np.cosh(x - x0))**2
operlin = -0.5 * Dxx + sp.diags(Vpot.flatten(), 0)

# Find eigenvalues and eigenvectors
lam, vec = spla.eigs(operlin, k=10, which='SM')
ll = np.diag(lam)

mu = 0.5
fresid = lambda u: (-0.5 * Dxx @ u.reshape(-1, 1) - u.reshape(-1, 1)**3 + Vpot * u.reshape(-1, 1) + mu * u.reshape(-1, 1)).flatten()

xStart = 2.5
uguess = (1 / np.cosh(x - xStart)).flatten()  # Initial guess, flattened
sol = fsolve(fresid, uguess).reshape(-1, 1)

plt.figure()
plt.plot(x, sol)
plt.title('Solution')
plt.show()

params = {'Dxx': Dxx, 'Vpot': Vpot}

# Solve ODE
density = sol * np.conj(sol)  # Equivalent to |u|^2
plt.plot(x, density)
plt.show()
print(f"density.shape = {density.shape}, x.shape = {x.shape}")
print(f"(x * density.real).shape = {(x * density.real).shape}")
numerator = np.trapz((x * density.real).flatten(), x.flatten())
print(f"numerator = {numerator}")
denominator = np.trapz(density.real.flatten(), x.flatten())
xmax = numerator / denominator

print(f"x.flatten().shape = {x.flatten().shape}, sol.flatten().shape = {sol.flatten().shape}")
print(f"x.reshape(-1,1).flatten().shape = {x.reshape(-1,1).flatten().shape}, sol.flatten().shape = {sol.flatten().shape}")
interpolator = interp1d(x.reshape(-1), sol.reshape(-1), kind='cubic', bounds_error=False, fill_value=np.nan)
print(f"sol.shape = {sol.shape}")
print(f"xmax = {xmax}")
sol = np.nan_to_num(interpolator(x.flatten() + (xmax - xStart)).reshape(-1, 1))
plt.plot(x, sol)
plt.axline((xStart, 0), (xStart, max(sol.flatten())))
plt.title('Solution')
plt.show()
print(f"sol.shape = {sol.shape}")
uinit = np.vstack([np.roll(sol, 1).reshape(-1, 1), np.zeros((nx, 1))])
print(f"uinit.shape = {uinit.shape}")
tspan = np.linspace(0, 100, 100001)
sol_ode = solve_ivp(lambda t, u: rate(t, u, params), [tspan[0], tspan[-1]], uinit.flatten(), t_eval=tspan, method="DOP853")

field = sol_ode.y[:nx, :] + 1j * sol_ode.y[nx:, :]
plt.figure()
plt.imshow(np.abs(field)**2, extent=[x.min(), x.max(), tspan.min(), tspan.max()], aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Field Evolution')
plt.show()

# Compute center of mass
com = np.zeros(len(tspan))
for i in range(len(tspan)):
    temp = field[:, i].reshape(-1)  # Flatten temp to shape (1201,)
    com[i] = np.trapz(x.flatten() * np.abs(temp)**2, x.flatten()) / np.trapz(np.abs(temp)**2, x.flatten())

plt.figure()
plt.plot(com)
plt.title('Center of Mass')
plt.show()
