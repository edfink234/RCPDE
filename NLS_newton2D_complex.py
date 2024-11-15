import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Parameters
w = 0.5  # Temporal frequency of sought steady state
g = 1   # g = 1 is defocusing, g = -1 is focusing
L, R = -10, 10  # Bounds for x and y
Nx, Ny = 101, 101  # Number of points in x and y
x = np.linspace(L, R, Nx)
y = np.linspace(L, R, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

# 2D meshgrid
X, Y = np.meshgrid(x, y, indexing='ij')

# Discrete 2D Laplacian with periodic boundary conditions
ONE_x = np.ones(Nx)
ONE_y = np.ones(Ny)
D2x = sp.diags([ONE_x, -2 * ONE_x, ONE_x], offsets=[-1, 0, 1], shape=(Nx, Nx), format='lil')
#D2x[0, -1] = 1
#D2x[-1, 0] = 1
D2x /= dx**2

D2y = sp.diags([ONE_y, -2 * ONE_y, ONE_y], offsets=[-1, 0, 1], shape=(Ny, Ny), format='lil')
#D2y[0, -1] = 1
#D2y[-1, 0] = 1
D2y /= dy**2

D2 = sp.kron(sp.eye(Ny), D2x) + sp.kron(D2y, sp.eye(Nx))  # 2D Laplacian

# Indices for real and imaginary parts
Nxy = Nx * Ny
indR = np.arange(Nxy)
indI = np.arange(Nxy, 2 * Nxy)

# Initial guess
A = np.sqrt(2 * w)
x0, y0 = 0, 0  # Center of initial guess
u0 = A * np.tanh(np.sqrt((X - x0)**2 + (Y - y0)**2))**2  # 2D tanh soliton

# Perturbation
np.random.seed(0)
pert = 0.1
up = u0 + pert * (np.random.rand(Nx, Ny) - 0.5) * np.exp(-((X**2 + Y**2) / 10))
U = np.hstack([up.real.ravel(), np.zeros_like(up.real).ravel()])  # Initial guess

# Potential V (set to zero if not defined)
V = np.zeros((Nx, Ny)).ravel()

# Newton's method
it = 0
err = np.inf
tol = 1e-10
alpha = 1

# Set up the initial plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, U[indR].reshape(Nx, Ny), cmap="viridis")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('Re(u)')
ax.set_title(f'Iteration = 0, Error = {err:.2e}')

# Set up the boundary condition enforcement
def apply_boundary_condition(U, J, F, boundary_value=1.0):
    """Apply boundary conditions to U at the boundary indices."""
    pass


# Iteration loop
while err > tol:
    it += 1
    Ur = U[indR].reshape(Nx, Ny)
    Ui = U[indI].reshape(Nx, Ny)

    # Modulus squared of u
    U2 = Ur**2 + Ui**2

    # Jacobian matrix components
    J11 = -0.5 * D2 + sp.diags(g * (3 * Ur.ravel()**2 + Ui.ravel()**2) + V + w)
    J22 = -0.5 * D2 + sp.diags(g * (Ur.ravel()**2 + 3 * Ui.ravel()**2) + V + w)
    J12 = sp.diags(2 * g * Ur.ravel() * Ui.ravel())

    # Full Jacobian matrix
    J = sp.bmat([[J11, J12], [J12, J22]], format='csr')

    # RHS of the system
    Fr = (-0.5 * D2 @ Ur.ravel() + (g * U2.ravel() + V + w) * Ur.ravel())
    Fi = (-0.5 * D2 @ Ui.ravel() + (g * U2.ravel() + V + w) * Ui.ravel())
    F = np.hstack([Fr, Fi])
    
    apply_boundary_condition(U, J, F, boundary_value=1.0)

    # Newton correction
    DU = spla.spsolve(J, -F)
    U1 = U + alpha * DU
    err = np.linalg.norm(F)
    print("err =", err)
#    if err < 8.2e-6:
#        alpha *= 0.5
    
    # Update U for the next iteration
    U = U1

    # Update the surface plot data instead of clearing the figure
    surf.remove()  # Remove the previous surface plot
    surf = ax.plot_surface(X, Y, Ur, cmap="viridis")  # Redraw the updated surface
    ax.set_title(f'Iteration = {it}, Error = {err:.2e}')
    ax.set_zlim(0,1)
    plt.draw()
    plt.pause(2)

plt.close()

## Final plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Ur, cmap="viridis")
ax.set_title(f'Final, Error = {err:.2e}')
plt.show()
