# Numerically integrate heat equation in 2D:
# u_t = (i/2)*u_xx - i*g*(|u|^2)*u
# using finite differences in space and RK4 in time
# RCG Jul'24

from math import pi
from numpy import linspace, sin, array, arange, exp, sin, meshgrid, min, max, arctan, sqrt, cosh, conj, real, nan, any, isnan, logical_not, nan_to_num
from matplotlib.cm import coolwarm
from matplotlib.pyplot import plot, legend, draw, pause, close, xlabel, ylabel, title, clf, xlim, ylim, show, subplots
import matplotlib.patches as mpatches

Lx = -10;      # left end of domain in x
Rx = +10;      # right end of domain in x
Ly = -10;      # left end of domain in y
Ry = +10;      # right end of domain in y
Nx = 100;      # number of points in x direction
Ny = 100;      # number of points in y direction
dt=0.0001;     # time step
tmax = 10;      # max time
g=-1;     # non-linearity (g=-1: attractive; g=+1: repulsive)

#Define discrete domain:
dx = (Rx - Lx) / Nx;
dy = (Ry - Ly) / Ny;
x = linspace(Lx,Rx-dx,Nx); # remove one dx to make solution periodic on [L,R]
y = linspace(Ly,Ry-dy,Ny); # remove one dx to make solution periodic on [L,R]
X, Y = meshgrid(x, y);

oodx2=1/dx**2;
oody2=1/dy**2;

sech = lambda x: 1/cosh(x)

#Define ICs:
A=1;          # amplitude for IC
v=1;          # initial vel
x0=0;         # initial pos
u0_x=A*sech(A*(X-x0))*exp(1j*v*X);
maxu0_x=abs(u0_x).max();

#Define ICs:
A=1;          # amplitude for IC
v=1;          # initial vel
y0=0;         # initial pos
u0_y=A*sech(A*(Y-y0))*exp(1j*v*Y);
maxu0_y=abs(u0_y).max();

#u0 = u0_x * u0_y;
X[X==0] = 1e-45; u0 = A*sech(A*sqrt(X**2+Y**2))*exp(1j*arctan(Y/X));
real_u0 = real(nan_to_num(u0, nan = 0))#real(u0[logical_not(isnan(u0))])

minu0 = real_u0.min()
maxu0 = real_u0.max()

myax = [Lx, Rx, Ly, Ry, minu0, maxu0];
#
#nifty indexing for u_xx:
ip_x = array([*range(2, Nx+1),1]) # ip=[2:Nx,1]; # periodic BCs
im_x = array([Nx, *range(1, Nx)]) # im=[Nx,1:Nx-1]; # periodic BCs
#shifting indices back 1 since index counting in Python starts at 0, not 1 like in Matlab
ip_x = ip_x - 1
im_x = im_x - 1

#nifty indexing for u_xx:
ip_y = array([*range(2, Ny+1),1]) # ip=[2:Ny,1]; # periodic BCs
im_y = array([Ny, *range(1, Ny)]) # im=[Ny,1:Ny-1]; # periodic BCs
#shifting indices back 1 since index counting in Python starts at 0, not 1 like in Matlab
ip_y = ip_y - 1
im_y = im_y - 1

fig, (ax1, ax2) = subplots(ncols = 2, subplot_kw={"projection": "3d"})
ax2.plot_surface(X, Y, real(u0), cmap = coolwarm)
pause(1e-45)
u=u0;

tmax += dt #np.arange goes up to but not including max in arange(min, max, step)

ii=0;
for t in arange(0, tmax, dt):
    ii += 1
    uxx1=oodx2*(u[:, im_x]-2*u+u[:, ip_x]);
    uyy1=oody2*(u[:, im_y]-2*u+u[:, ip_y]);
    k1 = 0.5j*(uxx1+uyy1) - 1j*g*(u**2)*conj(u)

    u_temp = u+0.5*dt*k1
    uxx2=oodx2*(u_temp[:, im_x]-2*u_temp+u_temp[:, ip_x]);
    uyy2=oody2*(u_temp[:, im_y]-2*u_temp+u_temp[:, ip_y]);
    k2 = 0.5j*(uxx2+uyy2) - 1j*g*(u**2)*conj(u)
    
    u_temp = u+0.5*dt*k2
    uxx3=oodx2*(u_temp[:, im_x]-2*u_temp+u_temp[:, ip_x]);
    uyy3=oody2*(u_temp[:, im_y]-2*u_temp+u_temp[:, ip_y]);
    k3 = 0.5j*(uxx3+uyy3) - 1j*g*(u**2)*conj(u)
    
    u_temp = u+dt*k3
    uxx4=oodx2*(u_temp[:, im_x]-2*u_temp+u_temp[:, ip_x]);
    uyy4=oody2*(u_temp[:, im_y]-2*u_temp+u_temp[:, ip_y]);
    k4 = 0.5j*(uxx4+uyy4) - 1j*g*(u**2)*conj(u)
    
    u = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    if ii % 1000 ==1:
    #    utheo = u0*exp(-(lambda_x + lambda_y)*D*t);

    #    ax1.cla()
        ax2.cla()
        
    #    surf1 = ax1.plot_surface(X, Y, utheo, cmap = coolwarm)
    #    ax1.set_title(f"Theoretical t={t:.2f}")
    #    ax1.set_xlim(*myax[:2])
    #    ax1.set_ylim(*myax[2:4])
    #    ax1.set_zlim(*myax[4:])
    #    ax1.set_xlabel('$x$');
    #    ax1.set_ylabel('$y$');
    #    ax1.set_zlabel('$u(x,y,t)$');
    
        surf2 = ax2.plot_surface(X, Y, real(u), cmap = coolwarm)

    #    fig.colorbar(surf, ax = ax2, shrink = 0.5, aspect = 5)
        ax2.set_title(f"Numerical t={t:.2f}")
        ax2.set_xlim(*myax[:2])
        ax2.set_ylim(*myax[2:4])
        ax2.set_zlim(*myax[4:])
        ax2.set_xlabel('$x$');
        ax2.set_ylabel('$y$');
        ax2.set_zlabel('$u(x,y,t)$');

        pause(1e-45)


#

