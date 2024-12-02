import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from numpy import tanh, exp, cos, sin, cosh, arccos, arccos as acos, log, sqrt
from os import system
from numpy.linalg import cond
from scipy.sparse.linalg import LinearOperator, spilu
from time import time
from matplotlib.colors import LightSource
from scipy.sparse import diags

sech = lambda x: 1/np.cosh(x)


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
    npts = npts_x * npts_y
    # Pre-allocate nonlinear residual
    resid = np.zeros(2 * npts)
    # Decompose the field into real and imaginary parts
    X = psi[:npts].reshape(npts_x, npts_y)
    Y = psi[npts:].reshape(npts_x, npts_y)

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
    
    # Compute corner terms using MSD boundary conditions
#    dens_tl = dens[1, 0]  # Top-left adjacent density
#    dens_tr = dens[1, -1]  # Top-right adjacent density
#    dens_bl = dens[-2, 0]  # Bottom-left adjacent density
#    dens_br = dens[-2, -1]  # Bottom-right adjacent density
#
#    Xdd_tl = 2 * (dens_tl - dens[0, 0] + V[1, 0] - V[0, 0]) * X[0, 0]
#    Xdd_tr = 2 * (dens_tr - dens[0, -1] + V[1, -1] - V[0, -1]) * X[0, -1]
#    Xdd_bl = 2 * (dens_bl - dens[-1, 0] + V[-2, 0] - V[-1, 0]) * X[-1, 0]
#    Xdd_br = 2 * (dens_br - dens[-1, -1] + V[-2, -1] - V[-1, -1]) * X[-1, -1]
#
#    Ydd_tl = 2 * (dens_tl - dens[0, 0] + V[1, 0] - V[0, 0]) * Y[0, 0]
#    Ydd_tr = 2 * (dens_tr - dens[0, -1] + V[1, -1] - V[0, -1]) * Y[0, -1]
#    Ydd_bl = 2 * (dens_bl - dens[-1, 0] + V[-2, 0] - V[-1, 0]) * Y[-1, 0]
#    Ydd_br = 2 * (dens_br - dens[-1, -1] + V[-2, -1] - V[-1, -1]) * Y[-1, -1]
#
#    # Assign computed values to the corners
#    d2X[0, 0] = Xdd_tl
#    d2X[0, -1] = Xdd_tr
#    d2X[-1, 0] = Xdd_bl
#    d2X[-1, -1] = Xdd_br
#
#    d2Y[0, 0] = Ydd_tl
#    d2Y[0, -1] = Ydd_tr
#    d2Y[-1, 0] = Ydd_bl
#    d2Y[-1, -1] = Ydd_br


    # Return the system of nonlinear equations
    resid[:npts_x * npts_y] = (-0.5 * d2X + comm * X).ravel()
    resid[npts_x * npts_y:] = (-0.5 * d2Y + comm * Y).ravel()

    return resid

save_path = "/Users/edwardfinkelstein/RCPDE/EdwardF/imgs/pdfs/NLS_MSD_pdfs/NLS_2D_MSD_pdfs/"
# Parameters
params = {
    'nls': {
        'npts': (101, 101),  # Number of grid points in x and y directions
        'dx': (10 - (-10)) / 100,  # Mesh size in x direction
        'dy': (10 - (-10)) / 100,  # Mesh size in y direction
        'V': 0,  # Potential
        'mu': 9,  # Temporal frequency = A^2
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
x_0, y_0 = 0.0, 0.0  # Initial guess location
dist = np.sqrt( ((x_grid-x_0)**2 + (y_grid-y_0)**2) )
u0 = A * np.tanh(A * dist) * np.exp(1j*np.arctan2(y_grid, x_grid))  # Initial guess
#u0 = A * np.tanh(A * 0.8 * dist) * np.exp(1j*np.arctan2((y_grid-y_0), (x_grid-x_0)))  # Initial guess

# Perturbed initial guess
np.random.seed(0)  # For reproducibility
pert = 0
up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
U = np.hstack([up.real.ravel(), up.imag.ravel()])

# Newton's method
it = 0
err = np.inf
tol = 1e-9

Ix = sp.eye(npts_x)
Iy = sp.eye(npts_y)
Dx = sp.diags([np.ones(npts_x - 1), -2 * np.ones(npts_x), np.ones(npts_x - 1)], [-1, 0, 1], shape=(npts_x, npts_x)) / params['nls']['dx']**2
Dy = sp.diags([np.ones(npts_y - 1), -2 * np.ones(npts_y), np.ones(npts_y - 1)], [-1, 0, 1], shape=(npts_y, npts_y)) / params['nls']['dy']**2
Laplacian = -0.5 * (sp.kron(Iy, Dx) + sp.kron(Dy, Ix))
V_flat = params['nls']['V'].ravel()

# Plot 1: Number of iterations vs. perturbation size
def Plot1():
    pert_range = np.linspace(0.1, 3, 10)
    iterations = []
    err = np.inf

    start=time()

    for pert in pert_range:
        np.random.seed(0)
        up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
        U = np.hstack([up.real.ravel(), up.imag.ravel()])
        
        it = 0
        err = np.inf
        
        while err > tol:
            print(f"Iteration {it}, Error: {err:.2e}")
            it += 1

            # Compute residual using nls2d_msd
            F = nls2d_msd(U, params)

            # Apply modulus-squared Dirichlet boundary conditions
            Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
            Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
            
        #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
            
            # Diagonal terms for J11 and J22
            dens = Ur**2 + Ui**2
            
            J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
            J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

            # Off-diagonal terms J12 and J21
            J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
            J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

            # Assemble the full Jacobian as a block matrix
            J = sp.bmat([[J11, J12], [J21, J22]], format='csc')

            # Newton correction
            
            DU, info = spla.cg(J, -F, tol=1e-10)
            assert(info==0)
    #        DU = spla.spsolve(J, -F)

            U1 = U + DU
            err = np.linalg.norm(F)
            

            # Update U
            U = U1
        print(f"Iteration {it}, Error: {err:.2e}")
        iterations.append(it)

    print(f"time taken = {time() - start}")

    plt.figure()
    plt.plot(pert_range, iterations, '-o')
    plt.xlabel(r'Perturbation Size $\epsilon$')
    plt.ylabel('Number of Iterations')
    plt.title(r'Iterations vs. Perturbation Size $\epsilon$')
    plt.yticks(np.arange(min(iterations), max(iterations)+1, 1))  # This sets the x-ticks as integers
    plt.savefig(f'{save_path}iterations_vs_perturbation_2D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}iterations_vs_perturbation_2D.pdf {save_path}iterations_vs_perturbation_2D.svg")
    system(f"open {save_path}iterations_vs_perturbation_2D.pdf")
    system(f"rm {save_path}iterations_vs_perturbation_2D.svg")

# Plot 2: Error vs. number of iterations for a fixed perturbation size
def Plot2():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
    U = np.hstack([up.real.ravel(), up.imag.ravel()])
    errors = []

    while err > tol:
        print(f"Iteration {it}, Error: {err:.2e}")
        it += 1

        # Compute residual using nls2d_msd
        F = nls2d_msd(U, params)

        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
        Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
        
    #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
        
        # Diagonal terms for J11 and J22
        dens = Ur**2 + Ui**2
        
        J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
        J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

        # Off-diagonal terms J12 and J21
        J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
        J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

        # Assemble the full Jacobian as a block matrix
        J = sp.bmat([[J11, J12], [J21, J22]], format='csc')


        # Newton correction
        
        DU, info = spla.cg(J, -F, tol=1e-10)
        assert(info==0)
    #        DU = spla.spsolve(J, -F)

        U1 = U + DU
        err = np.linalg.norm(F)
        

        # Update U
        U = U1
        errors.append(err)

    plt.figure()
    plt.plot(errors, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(r'Error vs. Iterations for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 10))  # This sets the x-ticks as integers

    plt.savefig(f'{save_path}error_vs_iterations_2D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}error_vs_iterations_2D.pdf {save_path}error_vs_iterations_2D.svg")
    system(f"open {save_path}error_vs_iterations_2D.pdf")
    system(f"rm {save_path}error_vs_iterations_2D.svg")

# Plot 3: Solution evolution vs. number of iterations
def Plot3():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
    U = np.hstack([up.real.ravel(), up.imag.ravel()])
    solutions = [(np.abs(U[:npts_x * npts_y] + 1j * U[npts_x * npts_y:])**2).reshape(npts_x, npts_y)]

    while err > tol:
        print(f"Iteration {it}, Error: {err:.2e}")
        it += 1

        # Compute residual using nls2d_msd
        F = nls2d_msd(U, params)

        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
        Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
        
    #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
        
        # Diagonal terms for J11 and J22
        dens = Ur**2 + Ui**2
        
        J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
        J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

        # Off-diagonal terms J12 and J21
        J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
        J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

        # Assemble the full Jacobian as a block matrix
        J = sp.bmat([[J11, J12], [J21, J22]], format='csc')


        # Newton correction
        
        DU, info = spla.cg(J, -F, tol=1e-10)
        assert(info==0)
    #        DU = spla.spsolve(J, -F)

        U1 = U + DU
        err = np.linalg.norm(F)
        

        # Update U
        U = U1
        solutions.append((np.abs(U[:npts_x * npts_y] + 1j * U[npts_x * npts_y:])**2).reshape(npts_x, npts_y))

    fig = plt.figure(figsize=(15.5, 5))  # Adjust figure size for 3 subplots

    # Select the three specific iterations: 0, 1, and len(solutions)-1
    indices_to_plot = [0, 1, len(solutions) - 1]
    length = len(indices_to_plot)
    # Define RGB colors for the three iterations
    default_colors = plt.cm.tab10.colors  # tab10 is the default color cycle in many versions of Matplotlib
    colors = default_colors[:len(indices_to_plot)]  # Get the first 3 colors
    ls = LightSource(azdeg=180, altdeg=45)

    # Create subplots
    for idx, i in enumerate(indices_to_plot):
        ax = fig.add_subplot(1, length, idx + 1, projection='3d')  # Create 3 subplots in a row
        
        # Create the facecolor array with RGB tuples
        face_color = np.empty((x_grid.shape[0], x_grid.shape[1], 3), dtype=float)  # (M, N, 3) for RGB
        for j in range(x_grid.shape[0]):
            for k in range(x_grid.shape[1]):
                face_color[j, k] = colors[idx]  # Assign the RGB color

        # Plot the surface with the specified facecolors
        ax.plot_surface(x_grid, y_grid, solutions[i], facecolors=face_color, linewidth=0, edgecolor='none')
        ax.set_title(f"Iteration {i}")  # Optional: Add a title for each subplot
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel(r'$|u|^2$')

    fig.suptitle('Solution Evolution $|u|^2$ for $\epsilon = 1$')
    #plt.tight_layout(rect=[0., 0, 1, 1])  # Adjust left and right margins

    plt.savefig(f'{save_path}solution_evolution_2D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}solution_evolution_2D.pdf {save_path}solution_evolution_2D.svg")
    system(f"open {save_path}solution_evolution_2D.pdf")
    system(f"rm {save_path}solution_evolution_2D.svg")

    system(f"ls {save_path}")

# Plot 4: Physics guess vs SR guess
def Plot4():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    params['nls']['mu'] = 1
    A = params['nls']['mu']**0.5
    
    #Physics guess
    u0_ = A * np.tanh(A * dist) * np.exp(1j*np.arctan2((y_grid-y_0), (x_grid-x_0)))  # Initial guess
    up = u0_ + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
    U = np.hstack([up.real.ravel(), up.imag.ravel()])
    errors = []

    while err > tol:
        print(f"Iteration {it}, Error: {err:.2e}")
        it += 1

        # Compute residual using nls2d_msd
        F = nls2d_msd(U, params)

        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
        Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
        
    #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
        
        # Diagonal terms for J11 and J22
        dens = Ur**2 + Ui**2
        
        J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
        J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

        # Off-diagonal terms J12 and J21
        J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
        J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

        # Assemble the full Jacobian as a block matrix
        J = sp.bmat([[J11, J12], [J21, J22]], format='csc')

        # Newton correction
        
        DU, info = spla.cg(J, -F, tol=1e-10)
        assert(info==0)
    #        DU = spla.spsolve(J, -F)

        U1 = U + DU
        err = np.linalg.norm(F)
        

        # Update U
        U = U1
        errors.append(err)

    plt.figure()
    plt.plot(errors, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(r'Error vs. Iterations for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 30))  # This sets the x-ticks as integers

    plt.show()
    plt.close()
    
#    plt.savefig(f'{save_path}error_vs_iterations_2D.svg')
#    system(f"rsvg-convert -f pdf -o {save_path}error_vs_iterations_2D.pdf {save_path}error_vs_iterations_2D.svg")
#    system(f"open {save_path}error_vs_iterations_2D.pdf")
#    system(f"rm {save_path}error_vs_iterations_2D.svg")

    #SR guess
    
    it = 0
    err = np.inf
    u0_ = A * (tanh(dist) * sqrt(sech(sqrt(sech(-(-(dist))))))) * np.exp(1j*np.arctan2((y_grid-y_0), (x_grid-x_0)))  # Initial guess
    up = u0_ + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
    U = np.hstack([up.real.ravel(), up.imag.ravel()])
    errors = []

    while err > tol:
        print(f"Iteration {it}, Error: {err:.2e}")
        it += 1

        # Compute residual using nls2d_msd
        F = nls2d_msd(U, params)

        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
        Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
        
    #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
        
        # Diagonal terms for J11 and J22
        dens = Ur**2 + Ui**2
        
        J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
        J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

        # Off-diagonal terms J12 and J21
        J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
        J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

        # Assemble the full Jacobian as a block matrix
        J = sp.bmat([[J11, J12], [J21, J22]], format='csc')


        # Newton correction
        
        DU, info = spla.cg(J, -F, tol=1e-10)
        assert(info==0)
    #        DU = spla.spsolve(J, -F)

        U1 = U + DU
        err = np.linalg.norm(F)
        

        # Update U
        U = U1
        errors.append(err)

    plt.figure()
    plt.plot(errors, '-o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title(r'Error vs. Iterations for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 30))  # This sets the x-ticks as integers

    plt.show()
    
#    plt.savefig(f'{save_path}error_vs_iterations_2D.svg')
#    system(f"rsvg-convert -f pdf -o {save_path}error_vs_iterations_2D.pdf {save_path}error_vs_iterations_2D.svg")
#    system(f"open {save_path}error_vs_iterations_2D.pdf")
#    system(f"rm {save_path}error_vs_iterations_2D.svg")

# Plot 5: Error vs. number of iterations for a fixed perturbation size
def Plot5():
    def diagonal_preconditioner(matrix):
        diag = matrix.diagonal()
        inv_diag = 1.0 / diag
        return diags(inv_diag)
        
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts_x, npts_y) - 0.5) * np.exp(-(x_grid**2 + y_grid**2) / 10)
    U = np.hstack([up.real.ravel(), up.imag.ravel()])
    errors = []
    condition_numbers = []

    while err > tol:
        print(f"Iteration {it}, Error: {err:.2e}")
        it += 1

        # Compute residual using nls2d_msd
        F = nls2d_msd(U, params)

        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts_x * npts_y].reshape(npts_x, npts_y)
        Ui = U[npts_x * npts_y:].reshape(npts_x, npts_y)
        
    #    print(f"Ix.shape = {Ix.shape}, Iy.shape = {Iy.shape}, Dx.shape = {Dx.shape}, Dy.shape = {Dy.shape}, Laplacian.shape = {Laplacian.shape}")
        
        # Diagonal terms for J11 and J22
        dens = Ur**2 + Ui**2
        
        J11 = Laplacian + sp.diags(3 * Ur.ravel()**2 + Ui.ravel()**2 + V_flat - params['nls']['mu'])
        J22 = Laplacian + sp.diags(3 * Ui.ravel()**2 + Ur.ravel()**2 + V_flat - params['nls']['mu'])

        # Off-diagonal terms J12 and J21
        J12 = sp.diags(2 * Ur.ravel() * Ui.ravel())
        J21 = sp.diags(2 * Ur.ravel() * Ui.ravel())

        # Assemble the full Jacobian as a block matrix
        J = sp.bmat([[J11, J12], [J21, J22]], format='csc')

        ew1, ev = spla.eigsh(J, which='LM')
        ew2, ev = spla.eigsh(J, sigma=1e-8)   #<--- takes a long time

        ew1 = abs(ew1)
        ew2 = abs(ew2)

        condA = ew1.max()/ew2.min()
        
        condition_numbers.append(condA)
        # Newton correction
        
        DU, info = spla.cg(J, -F, tol=1e-10)
        assert(info==0)
    #        DU = spla.spsolve(J, -F)
        U1 = U + DU
        err = np.linalg.norm(F)
        

        # Update U
        U = U1
        errors.append(err)

    plt.figure()
    
    e_k_plus_1 = np.array(errors[1:])
    e_k = np.array(errors[:-1])
    ratio = e_k_plus_1/(e_k**2)
    plt.plot(ratio, '-o')
    plt.xlabel('Iteration')
    plt.ylabel(r'$e_{k+1}/e_{k}$')
    plt.title(r'Error ratio vs. Iterations for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 10))  # This sets the x-ticks as integers

    plt.savefig(f'{save_path}rel_error_vs_iterations_2D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}rel_error_vs_iterations_2D.pdf {save_path}rel_error_vs_iterations_2D.svg")
    system(f"open {save_path}rel_error_vs_iterations_2D.pdf")
    system(f"rm {save_path}rel_error_vs_iterations_2D.svg")
    plt.close()
    
    
    plt.plot(condition_numbers, '-o')
    plt.xlabel('Iteration')
    plt.ylabel(r'Condition Number of Jacobian')
    plt.title(r'Evolution of cond(J) for $\epsilon = 1$')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 10))  # This sets the x-ticks as integers

    plt.savefig(f'{save_path}Jac_condition_number_2D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}Jac_condition_number_2D.pdf {save_path}Jac_condition_number_2D.svg")
    system(f"open {save_path}Jac_condition_number_2D.pdf")
    system(f"rm {save_path}Jac_condition_number_2D.svg")
    

#Plot1()
#Plot2()
#Plot3()
#Plot4()
Plot5()


