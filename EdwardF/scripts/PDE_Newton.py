import numpy as np
import torch
from torch import diag
from warnings import filterwarnings
filterwarnings('ignore')

def sech(x):
    if isinstance(x, torch.Tensor):
        return 1 / torch.cosh(x)
    elif isinstance(x, (float, np.float64, int)):
        return 1 / np.cosh(x)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")

def potential(x, xi):
    V_MT = 0.5 * Omega * Omega * x * x
    temp = torch.sech(b * (x - xi))
    V_SECH = A * temp * temp
#    return V_MT + V_SECH
    return V_SECH

#    return torch.zeros_like(x)

torch.sech = sech
b = 1
A = 1
Omega = 0.2
dt = 0.001      # Time step
tol = 1e-10
g = -1;                         # g = 1 is defocusing and g =-1 is focusing
N = 400;                        # number of mesh points
L = -10; R = 10;                # Left and right bounds of interval
x = torch.linspace(L, R, N)[1:];    # discrete space
dx = x[1]-x[0];                 # mesh size
point_5_over_dx_squared = (0.5 / dx**2)
one_over_six = 1.0/6.0
print(np.sqrt(2)*dx*dx*0.5)
assert(dt<np.sqrt(2)*dx*dx*0.5)

A_sol = 2; c = 0;
w_sol = A_sol*A_sol*0.5 #temporal freq
u0 = A_sol*sech(A_sol*(x))*torch.exp(1j*c*x);    # initial condition (IC)
V = potential(x, 0)
U = torch.cat([torch.real(u0), torch.imag(u0)], dim=0)
print(U.shape, u0.shape, V.shape)

err = np.inf

# Define the discrete Laplacian with periodic boundary conditions
D2 = torch.zeros(N, N)

# Fill the main diagonal
D2 += torch.diag(-2 * torch.ones(N))

# Fill the off-diagonals
D2 += torch.diag(torch.ones(N - 1), diagonal=-1)
D2 += torch.diag(torch.ones(N - 1), diagonal=1)
D2[0, -1] = 1
D2[-1, 0] = 1
D2 = D2 / (dx**2)
print(D2)

# Index for real and imaginary parts
indR = slice(0, N)  # $u_{\mathrm{real}}$indices
indI = slice(N, 2 * N)  # $u_{\mathrm{imag}}$ part indices

u = U[indR]+1j*U[indI];      # wrapping into a complex vector
print(f"u.shape = {u.shape}")
print(f"Max(abs(u)) = {max(abs(u))}")
idx=np.where(np.isclose(abs(u), max(abs(u))))
print(f"idx = {idx}")
print(f"abs(u)[idx] = {abs(u)[idx]}")
print(f"x[idx] = {x[idx]}")
u_before = u.detach().clone()

# Main loop: checking Newton tolerance
num_iter = 0
while err > tol:
    # Split real and imaginary parts
    Ur = U[indR]
    Ui = U[indI]

    # Compute modulus squared of u
    U2 = Ur**2 + Ui**2

    # Jacobian components
    J11 = -0.5 * D2 + diag(g * (3 * Ur**2 + Ui**2) + V + w_sol)
    J22 = -0.5 * D2 + diag(g * (Ur**2 + 3 * Ui**2) + V + w_sol)
    J12 = g * diag(2 * Ur * Ui)
    J = torch.cat(
        [torch.cat([J11, J12], dim=1), torch.cat([J12, J22], dim=1)], dim=0
    )
    print(J)

    # Right-hand side (RHS)
    Fr = -0.5 * (D2 @ Ur) + (g * U2 + V + w_sol) * Ur
    Fi = -0.5 * (D2 @ Ui) + (g * U2 + V + w_sol) * Ui
    F = torch.cat([Fr, Fi], dim=0)

    # Newton correction
    DU = -torch.linalg.solve(J, F)  # Solve J * DU = -F
    U1 = U + DU  # Update solution
    print(f"DU.max() = {DU.max()}")

    # Update error and solution
    err = torch.norm(F).numpy().item()
    print(f"err = {err}")
    print(f"Condition number of J: {torch.linalg.cond(J)}")
    U = U1
    num_iter += 1




