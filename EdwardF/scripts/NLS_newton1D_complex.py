import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# Parameters
w = 0.5  # Temporal frequency of sought steady state
g = 1   # g = 1 is defocusing, g = -1 is focusing
L, R = -10, 10  # Left and right bounds of interval
Nx = 301  # Number of mesh points
x = np.linspace(L, R, Nx)  # Discrete space
dx = x[1] - x[0]  # Mesh size

alpha = 1.0  # Set desired |u(0)|^2 value
beta = 1.0   # Set desired |u(N-1)|^2 value

# Discrete Laplacian with periodic boundary conditions
ONE = np.ones(Nx)
D2 = sp.diags([ONE, -2 * ONE, ONE], offsets=[-1, 0, 1], shape=(Nx, Nx), format='csr')

'''
u_{x+1} + u_{x-1} - 2 u_{x}
'''
##
#D2[0, -1] = 1
#D2[-1, 0] = 1
D2 /= dx**2
#D2[0,0] = 1
#D2[0,1] = 0
#D2[1,0] = 0
#D2[-1,-1] = 1
#D2[-1, -2] = 0
#D2[-2, -1] = 0

print(D2.toarray())
#exit()
#D2[
# Indices for real and imaginary parts
indR = np.arange(Nx)
indI = np.arange(Nx, 2 * Nx)

# Initial guess
A = np.sqrt(2 * w)
x0 = (R + L) / 2  # Amplitude and position of initial guess
u0 = -A * np.abs(np.tanh(A * (x - x0)))  # Unperturbed initial guess (sech soliton)
#u0 = A * np.cosh(A * (x - x0))**-1  # Unperturbed initial guess (sech soliton)

# Perturbation
np.random.seed(0)  # Reset random generator for reproducibility
pert = 0  # Size of perturbation
up = u0 + pert * (np.random.rand(Nx) - 0.5) * np.exp(-x**2 / 10)  # Perturbed initial condition
U = np.hstack([up.real, up.imag])  # Initial guess with real and imaginary parts
print(U.shape)

# Potential V (set to zero if not defined)
#V = 0.5*(x**2) + 10*np.exp(-x**2)
V = 0
# Newton's method
it = 0
err = 1  # Initial error
tol = 1e-13  # Convergence tolerance

# Set modulus-squared Dirichlet boundary conditions
alpha, beta = 1.5, 1.5  # Desired |u(0)|^2 = alpha and |u(N-1)|^2 = beta

# Inside the Newton iteration loop:
while err > tol:
    it += 1
    Ur = U[indR]
    Ui = U[indI]

    # Modulus squared of u
    U2 = Ur**2 + Ui**2

    # Jacobian matrix components
    J11 = -0.5 * D2 + sp.diags(g * (3 * Ur**2 + Ui**2) + V + w)
    J22 = -0.5 * D2 + sp.diags(g * (Ur**2 + 3 * Ui**2) + V + w)
    J12 = sp.diags(2 * g * Ur * Ui)

    # Full Jacobian matrix
    J = sp.bmat([[J11, J12], [J12, J22]], format='csr')

    # RHS of the system
    Fr = -0.5 * D2 @ Ur + (g * U2 + V + w) * Ur
    Fi = -0.5 * D2 @ Ui + (g * U2 + V + w) * Ui
    F = np.hstack([Fr, Fi])  # Combine real and imaginary RHS

    # Apply modulus-squared Dirichlet boundary conditions
    F[0] = (Ur[0]**2 + Ui[0]**2)**.5 - alpha**.5  # Enforce |u(0)|^2 = alpha
#    F[Nx-1] = Ur[-1]**2 + Ui[-1]**2 - beta  # Enforce |u(N-1)|^2 = beta
#    F[Nx] = Ur[0]**2 + Ui[0]**2 - alpha  # Enforce |u(0)|^2 = alpha
    F[-1] = (Ur[-1]**2 + Ui[-1]**2)**.5 - beta**.5  # Enforce |u(N-1)|^2 = beta
    
    # Modify Jacobian for boundary conditions
    J[0, 0] = Ur[0]/(Ur[0]**2 + Ui[0]**2)**.5  # Real part at left boundary ∂F_0/∂U_r
    J[0, Nx] = Ui[0]/(Ur[0]**2 + Ui[0]**2)**.5  # Imaginary part at left boundary ∂F_0/∂U_i
    
    J[-1, Nx-1] = Ur[-1]/(Ur[-1]**2 + Ui[-1]**2)**.5  # Real part at right boundary ∂F_1/∂U_r
    J[-1, 2*Nx-1] = Ui[-1]/(Ur[-1]**2 + Ui[-1]**2)**.5  # Imaginary part at right boundary ∂F_1/∂U_i

    # Newton correction
    DU = spla.spsolve(J, -F)
    U1 = U + DU
    err = np.linalg.norm(F)  # Update error
    print(f"err={err:.2e}",end='\r',flush=True)

    # Plotting progress of iteration
#    plt.figure(1)
#    plt.plot(x, U[indR], '.', label="Re(previous)")
#    plt.plot(x, U[indI], '.', label="Im(previous)")
#    plt.plot(x, U1[indR], label="Re(current)")
#    plt.plot(x, U1[indI], label="Im(current)")
#    plt.plot(x, V * np.ones_like(x), label="V(x)")
#    plt.xlabel('x')
#    plt.ylabel('u')
#    #plt.ylim(0,1)
#    plt.title(f'Iteration = {it}, Error = {err:.2e}')
#    plt.legend(loc='upper right')
#    plt.draw()
#    plt.pause(1e-45)
#    plt.clf()

    # Update U for the next iteration
    U = U1

# Final solution
u = U[indR] + 1j * U[indI]  # Combine real and imaginary parts into a complex vector

# Final plot
plt.plot(x, u.real, label="Re(u)")
plt.plot(x, u.imag, label="Im(u)")
plt.xlabel('x')
plt.ylabel('u')
plt.title('Final Solution')
plt.legend()
plt.show()


