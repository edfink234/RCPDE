###################################################################
# NLS_newton1D_real.m:
# Using Newton's method to compute a REAL steady state for NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs, ones, arange, array, diag, block, hstack
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

A=sqrt(2*w);x0=(R+L)/2;  # Amplitude and position of initial guess
sech = lambda x: 1/cosh(x)
u0=A*sech(A*(x-x0));     # unperturbed initial guess (sech soliton)
seed(0)          # reset random generator
pert=1*0.01;            # size of perturbation
U=u0+pert*(rand(Nx)-0.5); # perturb IC = initial guess
it=0;err=1;             # initializing error

#Add potential
Omega=1*0.075;
V = Omega**2*x**2/2 + 0.30*sech(x/2)**2;

while(err>1e-12):        # Main loop: checking Newton tolerance
    it=it+1;

    U2 = U**2;
    J = -0.5*D2+diag(g*3*U2+V+w);   # Jacobian
    F = -0.5*D2*U+(g*U2+V+w)*U;    # RHS

    DU, *_ = lstsq(-J, F, rcond=-1)            # Newton correction
    U1 = U+DU;            # New step through Newton
    err=norm(F);          # How close to convergence we are

    plot(x, U, '.', x, U1,x,V,'-',x,u0,'-');
    title(f'it={it}, error={err:0.4e}')
    legend(['previous','current','V(x)','initial','Location','NE'])
    draw()
    pause(2)  # Allows the plot to update
    clf()  #

    U = U1;               # Update solution
u = U;
show()

