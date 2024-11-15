# Numerically integrate heat equation in 2D:
# u_t = D (u_xx + u_yy)
# using finite differences in space and Euler in time
# RCG Jul'24

from math import pi
from numpy import linspace, sin, array, arange, exp, sin, meshgrid, min, max
from matplotlib.cm import coolwarm
from matplotlib.pyplot import plot, legend, draw, pause, close, xlabel, ylabel, title, clf, xlim, ylim, show, subplots
import matplotlib.patches as mpatches

D = 1;         # diffusion coefficient
Lx = -10;      # left end of domain in x
Rx = +10;      # right end of domain in x
Ly = -10;      # left end of domain in y
Ry = +10;      # right end of domain in y
Nx = 100;      # number of points in x direction
Ny = 100;      # number of points in y direction
dt = 0.01;     # time step
tmax = 3;      # max time

#Define discrete domain:
dx = (Rx - Lx) / Nx;
dy = (Ry - Ly) / Ny;
x = linspace(Lx,Rx-dx,Nx); # remove one dx to make solution periodic on [L,R]
y = linspace(Ly,Ry-dy,Ny); # remove one dx to make solution periodic on [L,R]
X, Y = meshgrid(x, y);

oodx2=1/dx**2;
oody2=1/dy**2;

#Define ICs:
N0_x=3;         #number of periods in x
WL_x=(Rx-Lx)/N0_x;  # wavelength of IC in x
k0_x=2*pi/WL_x;   # wavenumber of IC
lambda_x=k0_x**2;  # exp decay rate for this IC
u0_x=sin(k0_x*X); # IC for x

N0_y=3;         #number of periods in y
WL_y=(Ry-Ly)/N0_y;  # wavelength of IC in y
k0_y=2*pi/WL_y;   # wavenumber of IC
lambda_y=k0_y**2;  # exp decay rate for this IC
u0_y=sin(k0_y*Y); # IC for y

u0 = u0_x * u0_y;

minu0 = min(u0);
maxu0 = max(u0);
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
ax2.plot_surface(X, Y, u0, cmap = coolwarm)
pause(1e-45)
u=u0;

tmax += dt #np.arange goes up to but not including max in arange(min, max, step)

for t in arange(0, tmax, dt):
    uxx=oodx2*(u[:, im_x]-2*u+u[:, ip_x]);
    uyy=oody2*(u[:, im_y]-2*u+u[:, ip_y]);

    RHS = D * (uxx + uyy); #+g*u.^2.*conj(u);
    u = u + dt*RHS; # Euler step

    utheo = u0*exp(-(lambda_x + lambda_y)*D*t);

    ax1.cla()
    ax2.cla()
    
    surf1 = ax1.plot_surface(X, Y, utheo, cmap = coolwarm)
    ax1.set_title(f"Theoretical t={t:.2f}")
    ax1.set_xlim(*myax[:2])
    ax1.set_ylim(*myax[2:4])
    ax1.set_zlim(*myax[4:])
    ax1.set_xlabel('$x$');
    ax1.set_ylabel('$y$');
    ax1.set_zlabel('$u(x,y,t)$');

    surf2 = ax2.plot_surface(X, Y, u, cmap = coolwarm)

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
