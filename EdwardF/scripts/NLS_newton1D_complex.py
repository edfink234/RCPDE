import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from os import system

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

save_path = "/Users/edwardfinkelstein/RCPDE/EdwardF/imgs/pdfs/NLS_MSD_pdfs/NLS_1D_MSD_pdfs/"
# Parameters
params = {
    'nls': {
        'npts': 301,  # Number of grid points
        'dx': (10 - (-10)) / 300,  # Mesh size
        'V': 0,  # Potential
        'mu': 9,  # Temporal frequency = A^2
    }
}
npts = params['nls']['npts']
params['nls']['V'] = np.zeros(npts)


# Spatial grid
L, R = -10, 10
x = np.linspace(L, R, npts)

# Initial guess
w = params['nls']['mu']
A = np.sqrt(w)
x0 = (R + L) / 2  # Center of initial guess
u0 = A * np.tanh(A * (x - x0))  # Initial guess (tanh soliton)

# Perturbed initial guess
np.random.seed(0)  # For reproducibility
pert = 2
up = u0 + pert * (np.random.rand(npts) - 0.5) * np.exp(-x**2 / 10)
U = np.hstack([up.real, up.imag])  # Real and imaginary parts concatenated
tol = 1e-9

# Plot 1: Number of iterations vs. perturbation size
def Plot1():
    pert_range = np.linspace(0.1, 3, 10)
    iterations = []
    err = np.inf

    for pert in pert_range:
        np.random.seed(0)
        up = u0 + pert * (np.random.rand(npts) - 0.5) * np.exp(-x**2 / 10)
        U = np.hstack([up.real, up.imag])
        
        it = 0
        err = np.inf
        
        while err > tol:
            it += 1
            
            # Compute residual using nls1d_msd
            F = nls1d_msd(U, params)
            
            # Apply modulus-squared Dirichlet boundary conditions
            Ur = U[:npts]
            Ui = U[npts:]
            
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
            
            # Newton correction
            DU = spla.spsolve(J, -F)
            U1 = U + DU
            err = np.linalg.norm(F)
            print(f"Iteration {it}, Error: {err:.2e}")
            
            # Update U
            U = U1
        
        iterations.append(it)

    plt.figure()
    plt.plot(pert_range, iterations, '-o')
    plt.xlabel(r'Perturbation Size $\epsilon$')
    plt.ylabel('Number of Iterations')
    plt.title(r'Iterations vs. Perturbation Size $\epsilon$')
    plt.yticks(np.arange(min(iterations), max(iterations)+1, 1))  # This sets the x-ticks as integers
    plt.savefig(f'{save_path}iterations_vs_perturbation_1D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}iterations_vs_perturbation_1D.pdf {save_path}iterations_vs_perturbation_1D.svg")
    system(f"open {save_path}iterations_vs_perturbation_1D.pdf")
    system(f"rm {save_path}iterations_vs_perturbation_1D.svg")

# Plot 2: Error vs. number of iterations for a fixed perturbation size
def Plot2():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts) - 0.5) * np.exp(-x**2 / 10)
    U = np.hstack([up.real, up.imag])
    errors = []

    while err > tol:
        it += 1
        
        # Compute residual using nls1d_msd
        F = nls1d_msd(U, params)
        
        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts]
        Ui = U[npts:]
        
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
        
        # Newton correction
        DU = spla.spsolve(J, -F)
        U1 = U + DU
        err = np.linalg.norm(F)
        print(f"Iteration {it}, Error: {err:.2e}")
        
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
    plt.xticks(np.arange(0, len(errors), 1))  # This sets the x-ticks as integers
    plt.savefig(f'{save_path}error_vs_iterations_1D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}error_vs_iterations_1D.pdf {save_path}error_vs_iterations_1D.svg")
    system(f"open {save_path}error_vs_iterations_1D.pdf")
    system(f"rm {save_path}error_vs_iterations_1D.svg")


# Plot 3: Solution evolution vs. number of iterations
def Plot3():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts) - 0.5) * np.exp(-x**2 / 10)
    U = np.hstack([up.real, up.imag])

    solutions = [np.abs(U[:npts] + 1j * U[npts:])**2]
    while err > tol:
        it += 1
        
        # Compute residual using nls1d_msd
        F = nls1d_msd(U, params)
        
        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts]
        Ui = U[npts:]
        
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
        
        # Newton correction
        DU = spla.spsolve(J, -F)
        U1 = U + DU
        err = np.linalg.norm(F)
        print(f"Iteration {it}, Error: {err:.2e}")
        
        # Update U
        U = U1
        solutions.append(np.abs(U[:npts] + 1j * U[npts:])**2)

    # Select the three specific iterations: 0, len(solutions)//2, and len(solutions)-1
    indices_to_plot = [0, (len(solutions)-1) // 3, len(solutions) - 1]

    plt.figure()
    for i in indices_to_plot:
        plt.plot(x, solutions[i], label=f'Iteration {i}')
    plt.xlabel('x')
    plt.ylabel(r'$|u|^2$')
    plt.title('Solution Evolution $|u|^2$ for $\epsilon = 1$')
    plt.legend()

    plt.savefig(f'{save_path}solution_evolution_1D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}solution_evolution_1D.pdf {save_path}solution_evolution_1D.svg")
    system(f"open {save_path}solution_evolution_1D.pdf")
    system(f"rm {save_path}solution_evolution_1D.svg")

    system(f"ls {save_path}")

# Plot 4: Relative Error vs. number of iterations for a fixed perturbation size
def Plot4():
    pert = 1.0
    it = 0
    err = np.inf
    np.random.seed(0)
    up = u0 + pert * (np.random.rand(npts) - 0.5) * np.exp(-x**2 / 10)
    U = np.hstack([up.real, up.imag])
    errors = []
    condition_numbers = []

    while err > tol:
        it += 1
        
        # Compute residual using nls1d_msd
        F = nls1d_msd(U, params)
        
        # Apply modulus-squared Dirichlet boundary conditions
        Ur = U[:npts]
        Ui = U[npts:]
        
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
        
        ew1, ev = spla.eigsh(J, which='LM')
        ew2, ev = spla.eigsh(J, sigma=1e-8)   #<--- takes a long time

        ew1 = abs(ew1)
        ew2 = abs(ew2)

        condA = ew1.max()/ew2.min()
        print("condition number =",condA)
        condition_numbers.append(condA)
        
        # Newton correction
        DU = spla.spsolve(J, -F)
        U1 = U + DU
        err = np.linalg.norm(F)
        print(f"Iteration {it}, Error: {err:.2e}")
        
        # Update U
        U = U1
        errors.append(err)

    e_k_plus_1 = np.array(errors[1:])
    e_k = np.array(errors[:-1])
    plt.figure()
    plt.plot(e_k_plus_1/(e_k**2), '-o')
    plt.xlabel('Iteration')
    plt.ylabel(r'$e_{k+1}/e_{k}$')
    plt.title(r'Error ratio vs. Iterations for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 1))  # This sets the x-ticks as integers
    plt.savefig(f'{save_path}rel_error_vs_iterations_1D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}rel_error_vs_iterations_1D.pdf {save_path}rel_error_vs_iterations_1D.svg")
    system(f"open {save_path}rel_error_vs_iterations_1D.pdf")
    system(f"rm {save_path}rel_error_vs_iterations_1D.svg")
    
    plt.close()
    
    
    plt.plot(condition_numbers, '-o')
    plt.xlabel('Iteration')
    plt.ylabel(r'Condition Number of Jacobian')
    plt.title(r'Evolution of cond(J) for $\epsilon = 1$')
    plt.yscale('log')
    # Set x-axis ticks to integers
    plt.xticks(np.arange(0, len(errors), 10))  # This sets the x-ticks as integers

    plt.savefig(f'{save_path}Jac_condition_number_1D.svg')
    system(f"rsvg-convert -f pdf -o {save_path}Jac_condition_number_1D.pdf {save_path}Jac_condition_number_1D.svg")
    system(f"open {save_path}Jac_condition_number_1D.pdf")
    system(f"rm {save_path}Jac_condition_number_1D.svg")


#Plot1()
#Plot2()
#Plot3()
Plot4()
