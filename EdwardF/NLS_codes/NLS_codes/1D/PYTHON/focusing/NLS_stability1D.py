###################################################################
# NLS_stability1D.m:
# Compute spectrum for a steady state for NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs, ones, arange, array, diag, block, hstack, argsort, flipud, conj
from numpy.linalg import eig
from numpy.random import seed, rand
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, draw, pause, show, close, legend, clf
from scipy.sparse import spdiags, csr_matrix
from scipy.sparse.linalg import eigs
from ODE_RK4 import ODE_RK4

from NLS_newton1D_complex import *       # first find steady state using Newton
# If no potential just make it zero
try:
    V
except NameError:
    V=0;
    
# Jacobian:

M11 = -(-0.5*D2+diag(2*g*u*conj(u)+V+w));
M12 = -diag(g*u*u);
M21 = -conj(M12);
M22 = -conj(M11);
M=1j*block([[M11,M12], [M21,M22]]);

neigs=0;                 # if neigs>0 only compute neigs evals
if(neigs>0):             # neigs evals about z0
#    optionseigs.disp=0;
    z0 = 0.5+0*1j;
    eee,vvv = eigs(csr_matrix(M), k = neigs, sigma = z0, which = 'LM');
else:                    # Full spectrum
    eee, vvv = eig(M);

ee=eee;#diag(eee);                          # eigenvalues
vv = vvv[0:Nx,:] + conj(vvv[Nx:,:]);  # eigenvectors
bb = (argsort(real(ee))); # sort by largest real part
bb = flipud(bb);
ee = ee[bb];
vv = vv[:,bb];

plot(real(ee), imag(ee),'o')
xlabel(r'Re($\lambda$)')
ylabel(r'Im($\lambda$)')
show()
