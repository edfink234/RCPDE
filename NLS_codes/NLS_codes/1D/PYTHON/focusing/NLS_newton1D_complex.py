###################################################################
# NLS_newton1D_complex.m:
# Using Newton's method to compute a COMPLEX steady state for NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at:
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs, ones, arange, array, diag, block, hstack, zeros
from numpy.random import seed, rand
from numpy.linalg import lstsq, norm
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, draw, pause, show, close, legend, clf
from scipy.sparse import spdiags, csr_matrix
from ODE_RK4 import ODE_RK4

w =0.5;                 # temporal frequency of sought steady state
g =-1;                  # g = 1 is defocusing and g =-1 is focusing
L =-10; R=10;           # Left and right bounds of interval
Nx=301;                 # number of mesh points
x =linspace(L,R,Nx);   # discrete space
dx=x[1]-x[0];           # mesh size
ONE=ones(Nx)           # unit vector for the discrete Laplacian
D2=spdiags([ONE,-2*ONE,ONE], [-1, 0, 1], Nx, Nx).tocsc().toarray() # Discrete Laplacian
D2[0,-1]=1; D2[-1,0]=1;                  # Periodic Boundary Conds
D2=D2/(dx**2);
D2=csr_matrix(D2)
indR=arange(1,Nx+1); indI=arange(Nx+1,2*Nx+1);               # index for real & imag
#shifting indices back 1 since index counting in Python starts at 0, not 1 like in Matlab
indR = indR - 1
indI = indI - 1

A=sqrt(2*w);x0=(R+L)/2;  # Amplitude and position of initial guess
sech = lambda x: 1/cosh(x)
u0=A*sech(A*(x-x0));     # unperturbed initial guess (sech soliton)
seed(0)          # reset random generator
pert=0.3;                # size of perturbation
up=u0+pert*(rand(Nx)-0.5)*exp(-x**2/10); # perturb IC
#print(up.shape)
U=hstack((real(up), imag(up)));  # initial guess
#print(U.shape)

it=0; err=1;             # initializing error
# If no potential => V=0
try:
    V
except NameError:
    V=zeros(x.shape);

while(err>1e-10):            # Main loop: checking Newton tolerance
    it=it+1;
    Ur=U[indR]; Ui=U[indI];     # real and imag parts of u

    U2 = Ur**2+Ui**2;                          # mod square of u
    J11=-0.5*D2+diag(g*(3*Ur**2+ Ui**2)+V+w); # J11 part of Jacobian
    J22=-0.5*D2+diag(g*(Ur**2+3*Ui**2)+V+w); # J22 part of Jacobian
    J12=g*diag(2*Ur*Ui);                      # J12 part of Jacobian

    J = block([[J11, J12], [J12,J22]])        # Full Jacobian
    Fr = -0.5*D2*Ur+(g*U2+V+w)*Ur;            # real(RHS)
    Fi = -0.5*D2*Ui+(g*U2+V+w)*Ui;            # imag(RHS)
#    print(Fr.shape, Fi.shape, J.shape)

    F = hstack((Fr, Fi))                      # RHS
    DU, *_ = lstsq(-J, F, rcond=-1)           # Newton correction
    U1 = U+DU;                  # New step through Newton
    err=norm(F);                # How close to convergence we are

    plot(x,U[indR],'.',x,U[indI],'.',x,U1[indR],x,U1[indI],x,V);
    xlabel('x'); ylabel('u');

    title(f'it={it}, error={err:0.4e}')
    legend(['Re(previous)','Im(previous)','Re(current)','Im(current)','V(x)','Location','NE'])
    draw()
    pause(2)  # Allows the plot to update
    clf()  #
#
    U = U1;                     # Update solution

u = U[indR]+1j*U[indI];      # wrapping into a complex vector
show()
