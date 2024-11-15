# Numerically integrate NLS in 1D:
# u_t = (i/2)*u_xx - i*g*(|u|^2)*u
# using finite differences in space and Euler in time
# RCG Jul'24

from math import pi
from numpy import linspace, sin, array, arange, exp, cosh, abs, conj, full_like, real, imag
from numpy.random import random
from matplotlib.pyplot import plot, legend, draw, pause, close, xlabel, ylabel, title, clf, xlim, ylim, show

L=-10;   # left end of domain
R=+10;   # right end of domain
Nx=400;  # number of points in discrete domain
dt=0.00001; # time step
tmax=10;  # max time
g=-1;     # non-linearity (g=-1: attractive; g=+1: repulsive)

#Define discrete domain:
dx=(R-L)/Nx;
x=linspace(L,R-dx,Nx); # remove one dx to make solution periodic on [L,R]
oodx2=1/dx**2;

#Define ICs:
A=1;          # amplitude for IC
v=1;          # initial vel
x0=0;         # initial pos
sech = lambda x: 1/cosh(x)
u0=A*sech(A*(x-x0))*exp(1j*v*x);
maxu0=max(abs(u0));
myax=[L, R, -1.2*maxu0, 1.2*maxu0];

#nifty indexing for u_xx:
ip = array(list(range(2, Nx+1))+[1])  # ip=[2:Nx,1]; # periodic BCs
im = array([Nx] + list(range(1, Nx))) # im=[Nx,1:Nx-1]; # periodic BCs

#print(ip, im)

#shifting indices back 1 since index counting in Python starts at 0, not 1 like in Matlab
ip = ip - 1
im = im - 1

plot(x, abs(u0), "o-", label = r"$|u_0|$")
legend()
draw()
pause(1)
clf()

u=u0;

tmax += dt #np.arange goes up to but not including max in arange(min, max, step)
#print(u)
#print(u[im])
ii=0;

for t in arange(0, tmax, dt):
    ii+=1;
    uxx = oodx2*(u[im]-2*u+u[ip]);
    RHS = 0.5j*uxx - 1j*g*(u**2)*conj(u) #+g*u.^2.*conj(u);
    u = u + dt*RHS; # Euler step
    if ii%1000 == 1:
        utheo = A*sech(x-v*(t+dt)-x0)*exp(1j*v*x+0.5j*(A**2-v**2)*(t+dt));
        plot(x, u.real, "*", label = r"$\mathbb{R}e(u_{\mathrm{num}})$")
        plot(x, u.imag, ".", label = r"$\mathbb{I}m(u_{\mathrm{num}})$")
        plot(x, abs(u), "-", label = r"$|u_{\mathrm{num}}(x,t)|$")

        plot(x, utheo.real, "*", label = r"$\mathbb{R}e(u_{\mathrm{theo}})$")
        plot(x, utheo.imag, ".", label = r"$\mathbb{I}m(u_{\mathrm{theo}})$")
        plot(x, abs(utheo), "-", label = r"$|u_{\mathrm{theo}}(x,t)|$")

        xlim(*myax[:2])
        ylim(*myax[2:])

        xlabel(r'$x$');
        ylabel(r"$\mathbb{R}e(u)$" + ", " + r"$\mathbb{I}m(u)$" + ", " + r"$|u(x,t)|$");
        legend();
        title(f"NLS Equation RK4, t={t:0.3f}/{tmax-dt:0.3f}")
        draw()
        pause(1e-45)
        clf() if t != tmax-dt else show()
