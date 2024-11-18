import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def nls1d_msd(psi, params):
    """
    Compute the residual for the 1D Nonlinear Schrödinger equation with 
    modulus-squared Dirichlet boundary conditions.

    Parameters:
        psi (numpy.ndarray): Concatenated real and imaginary parts of the wave function.
        params (dict): Dictionary containing problem parameters.

    Returns:
        numpy.ndarray: Residual of the nonlinear system.
    """
    # Unpack parameters
    npts = params['nls']['npts']
    dx = params['nls']['dx']
    V = np.array(params['nls']['V'])
    mu = params['nls']['mu']
    
    # Pre-allocate nonlinear residual
    resid = np.zeros(2 * npts)
    
    # Decompose the field into real and imaginary parts
    X = psi[:npts]
    Y = psi[npts:]
    
    # Compute the density and common term
    dens = X**2 + Y**2
    comm = dens + V - mu * np.ones(npts)
    
    # Compute the 1D Laplacians inside the domain (finite differences)
    d2Xdx2 = np.diff(X, 2) / dx**2
    d2Ydx2 = np.diff(Y, 2) / dx**2
    
    # Compute common terms (Ω) at the boundaries
    term_l = (d2Xdx2[0] * X[1] + d2Ydx2[0] * Y[1]) / dens[1]
    term_r = (d2Xdx2[-1] * X[-2] + d2Ydx2[-1] * Y[-2]) / dens[-2]
    
    Omega_l = term_l - 2 * (dens[1] - dens[0] + V[1] - V[0]) #a = 1/2 in Ricardo's paper eq 3.4
    Omega_r = term_r - 2 * (dens[-2] - dens[-1] + V[-2] - V[-1]) #a = 1/2 in Ricardo's paper eq 3.4
    
    # Compute second-order derivatives at the endpoints
    Xdd_l = Omega_l * X[0]
    Xdd_r = Omega_r * X[-1]
    Ydd_l = Omega_l * Y[0]
    Ydd_r = Omega_r * Y[-1]
    
    # Update second derivatives with boundary values
    d2Xdx2 = np.concatenate(([Xdd_l], d2Xdx2, [Xdd_r]))
    d2Ydx2 = np.concatenate(([Ydd_l], d2Ydx2, [Ydd_r]))
    
    # Return the system of nonlinear equations
    resid[:npts] = -0.5 * d2Xdx2 + comm * X
    resid[npts:] = -0.5 * d2Ydx2 + comm * Y
    
    return resid


# Parameters
params = {
    'nls': {
        'npts': 301,  # Number of grid points
        'dx': (10 - (-10)) / 300,  # Mesh size
        'V': 0,  # Potential
        'mu': 1.4,  # Temporal frequency
        'g': 1,  # Defocusing parameter
    }
}
params['nls']['V'] = np.zeros(params['nls']['npts'])
Nx = params['nls']['npts']

# Spatial grid
L, R = -10, 10
x = np.linspace(L, R, params['nls']['npts'])

# Initial guess
w = params['nls']['mu']
A = 1.2#np.sqrt(2 * 1)
x0 = (R + L) / 2  # Center of initial guess
u0 = A * np.tanh(1 * (x - x0))  # Initial guess (sech soliton)

# Perturbed initial guess
np.random.seed(0)  # For reproducibility
pert = 2
up = u0 + pert * (np.random.rand(params['nls']['npts']) - 0.5) * np.exp(-x**2 / 10)
U = np.hstack([up.real, up.imag])  # Real and imaginary parts concatenated

# Newton's method
it = 0
err = 1
tol = 1e-9

while err > tol:
    it += 1
    
    # Compute residual using nls1d_msd
    F = nls1d_msd(U, params)
    
    # Apply modulus-squared Dirichlet boundary conditions
    Ur = U[:params['nls']['npts']]
    Ui = U[params['nls']['npts']:]
    npts = params['nls']['npts']
    dx = params['nls']['dx']
    V = np.array(params['nls']['V'])
    mu = params['nls']['mu']

    # Finite difference Laplacian for a 1D grid
    D2 = sp.diags([np.ones(npts), -2 * np.ones(npts), np.ones(npts)], [-1, 0, 1], shape=(npts, npts)) / dx**2

    # Diagonal terms for J11 and J22
    J11 = -0.5 * D2 + sp.diags(3*Ur**2 + Ui**2 + V - mu)  # Real part X
    J22 = -0.5 * D2 + sp.diags(3*Ui**2 + Ur**2 + V - mu)  # Imaginary part Y

    # Off-diagonal terms J12 and J21
    J12 = sp.diags(2 * Ur * Ui)
    J21 = sp.diags(2 * Ur * Ui)

    # Assemble the full Jacobian as a block matrix
    J = sp.bmat([[J11, J12], [J21, J22]], format='csr')

    
#    J[0, 0] = 2 * Ur[0]  # Real part at left boundary ∂F_0/∂U_r
#    J[0, Nx] = 2 * Ui[0]  # Imaginary part at left boundary ∂F_0/∂U_i
#
#    J[-1, Nx-1] = 2 * Ur[-1]  # Real part at right boundary ∂F_1/∂U_r
#    J[-1, 2*Nx-1] = 2 * Ui[-1]  # Imaginary part at right boundary ∂F_1/∂U_i
    
    # Newton correction
    DU = spla.spsolve(J, -F)
    U1 = U + DU
    err = np.linalg.norm(F)
    print(f"Iteration {it}, Error: {err:.2e}")
    
    # Update U
    U = U1

# Final solution
u = U[:params['nls']['npts']] + 1j * U[params['nls']['npts']:]

# Plot the final solution
plt.plot(x, u.real, label="Re(u)")
plt.plot(x, u.imag, label="Im(u)")
plt.xlabel('x')
plt.ylabel('u')
plt.title('Final Solution')
plt.legend()
plt.show()

