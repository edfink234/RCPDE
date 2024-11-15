###################################################################
# NLS_RHS.m : RHS for the NLS equation:
#    u_t = -i[-(1/2) u_xx + g*|u|^2 u + V(x)*u]
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################
#def NLS_RHS_2D(U,N,dx,g,V):
#    from numpy import hstack, vstack, conj, expand_dims
#
#    Upx = hstack((U[:,1:], expand_dims(U[:,0], 1)));            # periodic BCs
#    Umx = hstack((expand_dims(U[:,-1], 1), U[:,0:-1]));         # periodic BCs
#    Upy = vstack((U[1:,:], U[0,:]))             # periodic BCs
#    Umy = vstack((U[-1,:], U[0:-1,:]))          # periodic BCs
#
##    Up = hstack([0, U[:-1]]);         # Zero BCs
##    Um = hstack([*U[1:],0]);           # Zero BCs
##
##    Up = hstack((U[0]-(U[1]-U[0]), U[:-1]));      # linear extrapolation BCs
##    Um = hstack((U[1:], U[-1]-(U[-2]-U[-1])));   # linear extrapolation BCs
##
##    Up = hstack((U[0], U[:-1]));                  # Laplacian Zero BCs
##    Um = hstack((U[1:], U[-1]));                  # Laplacian Zero BCs
#
#    Uxx = (0.5/dx**2)*(Upx-2*U+Umx);
#    Uyy = (0.5/dx**2)*(Upy-2*U+Umy);
#
#    RHS = 1j*(Uxx + Uyy - (g*U*conj(U)+V)*U);
#
#    return RHS;

def NLS_RHS_2D(U,N,dx,g,V,mu):
    from numpy import hstack, vstack, conj, expand_dims

#    Upx = hstack((U[:,1:], expand_dims(U[:,0], 1)));            # periodic BCs
#    Umx = hstack((expand_dims(U[:,-1], 1), U[:,0:-1]));         # periodic BCs
#    Upy = vstack((U[1:,:], U[0,:]))             # periodic BCs
#    Umy = vstack((U[-1,:], U[0:-1,:]))          # periodic BCs

    Upx = hstack((U[:,1:], expand_dims(U[:,-2], 1)));            # Neumann BCs
    Umx = hstack((expand_dims(U[:,1], 1), U[:,0:-1]));         # Neumann BCs
    Upy = vstack((U[1:,:], U[-2,:]))             # Neumann BCs
    Umy = vstack((U[1,:], U[0:-1,:]))          # Neumann BCs


#    Up = hstack([0, U[:-1]]);         # Zero BCs
#    Um = hstack([*U[1:],0]);           # Zero BCs
#
#    Up = hstack((U[0]-(U[1]-U[0]), U[:-1]));      # linear extrapolation BCs
#    Um = hstack((U[1:], U[-1]-(U[-2]-U[-1])));   # linear extrapolation BCs
#
#    Up = hstack((U[0], U[:-1]));                  # Laplacian Zero BCs
#    Um = hstack((U[1:], U[-1]));                  # Laplacian Zero BCs

    Uxx = (0.5/dx**2)*(Upx-2*U+Umx);
    Uyy = (0.5/dx**2)*(Upy-2*U+Umy);

    RHS = 1j*(Uxx + Uyy - (g*U*conj(U)+V-mu)*U);

    return RHS;
