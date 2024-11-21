import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def nls2d_msd(psi, params):
    """
    Compute the residual for the 2D Nonlinear Schrödinger equation with
    modulus-squared Dirichlet boundary conditions.

    Parameters:
        psi (numpy.ndarray): Concatenated real and imaginary parts of the wave function.
        params (dict): Dictionary containing problem parameters.

    Returns:
        numpy.ndarray: Residual of the nonlinear system.
    """
    # Unpack parameters
    npts_x, npts_y = params['nls']['npts']
    dx, dy = params['nls']['dx'], params['nls']['dy']
    V = np.array(params['nls']['V']).reshape(npts_x, npts_y)
    mu = params['nls']['mu']

    # Pre-allocate nonlinear residual
    resid = np.zeros(2 * npts_x * npts_y)

    # Decompose the field into real and imaginary parts
    X = psi[:npts_x * npts_y].reshape(npts_x, npts_y)
    Y = psi[npts_x * npts_y:].reshape(npts_x, npts_y)

    # Compute the density and common term
    dens = X**2 + Y**2
    comm = dens + V - mu

    d2Xdx2 = np.diff(X, 2, axis=0) / dx**2
    d2Xdy2 = np.diff(X, 2, axis=1) / dy**2
    d2Ydx2 = np.diff(Y, 2, axis=0) / dx**2
    d2Ydy2 = np.diff(Y, 2, axis=1) / dy**2
    
    # Total second derivatives
    d2X = d2Xdx2[:, 1:-1] + d2Xdy2[1:-1, :]
    d2Y = d2Ydx2[:, 1:-1] + d2Ydy2[1:-1, :]
    # Now d2X and d2Y are (npts-2) x (npts-2) matrices
    # Now we have to add the boundary padding
    
    # Compute common terms (Ω) at the boundaries
    term_l = (d2X[:, 0] * X[1:-1, 1] + d2Y[:, 0] * Y[1:-1, 1]) / dens[1:-1, 1]
    term_r = (d2X[:, -1] * X[1:-1, -2] + d2Y[:, -1] * Y[1:-1, -2]) / dens[1:-1, -2]
    term_t = (d2X[0] * X[1, 1:-1] + d2Y[0] * Y[1, 1:-1]) / dens[1, 1:-1]
    term_b = (d2X[-1] * X[-2, 1:-1] + d2Y[-1] * Y[-2, 1:-1]) / dens[-2, 1:-1]
    
    Omega_l = term_l - 2 * (dens[1:-1, 1] - dens[1:-1, 0] + V[1:-1, 1] - V[1:-1, 0]) #a = 1/2 in Ricardo's paper eq 3.4
    Omega_r = term_r - 2 * (dens[1:-1, -2] - dens[1:-1, -1] + V[1:-1, -2] - V[1:-1, -1]) #a = 1/2 in Ricardo's paper eq 3.4
    Omega_t = term_t - 2 * (dens[1, 1:-1] - dens[0, 1:-1] + V[1, 1:-1] - V[0, 1:-1]) #a = 1/2 in Ricardo's paper eq 3.4
    Omega_b = term_b - 2 * (dens[-2, 1:-1] - dens[-1, 1:-1] + V[-2, 1:-1] - V[-1, 1:-1]) #a = 1/2 in Ricardo's paper eq 3.4
    
    # Compute second-order derivatives at the endpoints
    Xdd_l = Omega_l * X[1:-1, 0]
    Xdd_r = Omega_r * X[1:-1, -1]
    Xdd_t = Omega_t * X[0, 1:-1]
    Xdd_b = Omega_b *  X[-1, 1:-1]
    
    Ydd_l = Omega_l * Y[1:-1, 0]
    Ydd_r = Omega_r * Y[1:-1, -1]
    Ydd_t = Omega_t * Y[0, 1:-1]
    Ydd_b = Omega_b * Y[-1, 1:-1]
    
    d2X = np.pad(d2X, pad_width=1, mode='constant', constant_values=0)
    d2Y = np.pad(d2Y, pad_width=1, mode='constant', constant_values=0)
    
    # Update second derivatives with boundary values
    d2X[0, 1:-1] = Xdd_t
    d2X[-1, 1:-1] = Xdd_b
    d2X[1:-1, 0] = Xdd_l
    d2X[1:-1, -1] = Xdd_r
    
    d2Y[0, 1:-1] = Ydd_t
    d2Y[-1, 1:-1] = Ydd_b
    d2Y[1:-1, 0] = Ydd_l
    d2Y[1:-1, -1] = Ydd_r

    # Return the system of nonlinear equations
    resid[:npts_x * npts_y] = (-0.5 * d2X + comm * X).flatten()
    resid[npts_x * npts_y:] = (-0.5 * d2Y + comm * Y).flatten()

    return resid


# Parameters
params = {
    'nls': {
        'npts': (101, 101),  # Number of grid points in x and y directions
        'dx': (10 - (-10)) / 100,  # Mesh size in x direction
        'dy': (10 - (-10)) / 100,  # Mesh size in y direction
        'V': 0,  # Potential
        'mu': 2,  # Temporal frequency = A^2
    }
}
npts_x, npts_y = params['nls']['npts']
x = np.linspace(-10, 10, npts_x)
y = np.linspace(-10, 10, npts_y)
x_grid, y_grid = np.meshgrid(x, y, indexing='ij')
params['nls']['V'] = np.zeros((npts_x, npts_y))

# Initial guess
w = params['nls']['mu']
A = np.sqrt(w)
r0 = 0.0  # Radius for initial guess
u0 = A * np.tanh(A * np.sqrt((x_grid**2 + y_grid**2)) - r0) * np.exp(1j*np.arctan2(y_grid, x_grid))  # Initial guess

# Perturbed initial guess
np.random.seed(0)  # For reproducibility
pert = 2
up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
U = np.hstack([up.real.flatten(), up.imag.flatten()])

# Newton's method
it = 0
err = 1
tol = 1e-4

while err > tol:
    it += 1

    # Compute residual using nls2d_msd
    F = nls2d_msd(U, params)

    # Apply modulus-squared Dirichlet boundary conditions
    Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
    Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)

    # Laplacian operators for 2D grid
    Ix = sp.eye(npts_x)
    Iy = sp.eye(npts_y)
    Dx = sp.diags([np.ones(npts_x - 1), -2 * np.ones(npts_x), np.ones(npts_x - 1)], [-1, 0, 1], shape=(npts_x, npts_x)) / params['nls']['dx']**2
    Dy = sp.diags([np.ones(npts_y - 1), -2 * np.ones(npts_y), np.ones(npts_y - 1)], [-1, 0, 1], shape=(npts_y, npts_y)) / params['nls']['dy']**2
    Laplacian = sp.kron(Iy, Dx) + sp.kron(Dy, Ix)

    # Diagonal terms for J11 and J22
    dens = Ur**2 + Ui**2
    V_flat = params['nls']['V'].flatten()
    J11 = -0.5 * Laplacian + sp.diags(3 * Ur.flatten()**2 + Ui.flatten()**2 + V_flat - params['nls']['mu'])
    J22 = -0.5 * Laplacian + sp.diags(3 * Ui.flatten()**2 + Ur.flatten()**2 + V_flat - params['nls']['mu'])

    # Off-diagonal terms J12 and J21
    J12 = sp.diags(2 * Ur.flatten() * Ui.flatten())
    J21 = sp.diags(2 * Ur.flatten() * Ui.flatten())

    # Assemble the full Jacobian as a block matrix
    J = sp.bmat([[J11, J12], [J21, J22]], format='csr')

    # Newton correction
    DU = spla.spsolve(J, -F)
    U1 = U + DU
    err = np.linalg.norm(F)
    print(f"Iteration {it}, Error: {err:.2e}")

    # Update U
    U = U1

# Final solution
u = (U[:npts_x * npts_y] + 1j * U[npts_x * npts_y:]).reshape(npts_x, npts_y)

# Plot the final solution
# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(x_grid, y_grid, np.abs(u), cmap='viridis', edgecolor='none')

# Add a color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='|u|')

# Label the axes
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('|u|')

# Add a title
ax.set_title('Final Solution Magnitude')

plt.show()
