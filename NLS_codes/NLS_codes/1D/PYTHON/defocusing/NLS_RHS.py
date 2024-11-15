###################################################################
# NLS_RHS.m : RHS for the NLS equation:
#    u_t = -i[-(1/2) u_xx + g*|u|^2 u + V(x)*u]
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################
def NLS_RHS(u,N,dx,g,V):
    from numpy import hstack, conj

    up = hstack((u[1:], u[0]));       # periodic BCs
    um = hstack((u[-1], u[:-1]));      # periodic BCs
    
#    up = hstack((u[1:], u[-2]))
#    um = hstack((u[1], u[:-1]))

#    up = hstack([u[1:],0]);         # Zero (Dirichlet) BCs
#    um = hstack([0, u[:-1]]);           # Zero (Dirichlet) BCs
#
#    up = hstack((u[1:], u[-1]-(u[-2]-u[-1])));    # linear extrapolation BCs
#    um = hstack((u[0]-(u[1]-u[0]), u[:-1]));      # linear extrapolation BCs
#
#    up = hstack((u[1:], u[-1]));                    # Laplacian Zero BCs
#    um = hstack((u[0], u[:-1]));                    # Laplacian Zero BCs

    RHS = 1j*((0.5/(dx*dx))*(up-2*u+um) - (g*u*conj(u)+V)*u);

    return RHS;


