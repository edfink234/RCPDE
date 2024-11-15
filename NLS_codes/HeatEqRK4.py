# Numerically integrate heat equation in 1D:
# u_t = D u_xx
# using finite differences in space and RK4 in time
# RCG Jul'24

from math import pi
from numpy import linspace, sin, array, arange, exp
from matplotlib.pyplot import plot, legend, draw, pause, close, xlabel, ylabel, title, clf, xlim, ylim, show

D=1;     # diffusion coef
L=-10;   # left end of domain
R=+10;   # right end of domain
Nx=100;  # number of points in discrete domain
dt=0.02; # time step
tmax=3;  # max time

#Define discrete domain:
dx=(R-L)/Nx;
x=linspace(L,R-dx,Nx); # remove one dx to make solution periodic on [L,R]
oodx2=1/dx**2;

#Define ICs:
N0=3;         #number of periods in the domain
WL=(R-L)/N0;  # wavelength of IC
k0=2*pi/WL;   # wavenumber of IC
lambda_=k0**2;  # exp decay rate for this IC
u0=sin(k0*x); # IC
minu0=min(u0);
maxu0=max(u0);
myax=[L, R, minu0, maxu0];

#nifty indexing for u_xx:
ip = array(list(range(2, Nx+1))+[1])  # ip=[2:Nx,1]; # periodic BCs
im = array([Nx] + list(range(1, Nx))) # im=[Nx,1:Nx-1]; # periodic BCs

#print(ip, im)

#shifting indices back 1 since index counting in Python starts at 0, not 1 like in Matlab
ip = ip - 1
im = im - 1

plot(x, u0, "o-", label = r"$u_0$")
legend()
draw()
pause(1)
clf()

u=u0;

tmax += dt #np.arange goes up to but not including max in arange(min, max, step)

for t in arange(0, tmax, dt):
    uxx1 = oodx2 * (u[im] - 2 * u + u[ip])
    k1 = D * uxx1

    u_temp = u + 0.5 * dt * k1
    uxx2 = oodx2 * (u_temp[im] - 2 * u_temp + u_temp[ip])
    k2 = D * uxx2

    u_temp = u + 0.5 * dt * k2
    uxx3 = oodx2 * (u_temp[im] - 2 * u_temp + u_temp[ip])
    k3 = D * uxx3

    u_temp = u + dt * k3
    uxx4 = oodx2 * (u_temp[im] - 2 * u_temp + u_temp[ip])
    k4 = D * uxx4

    u = u + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    utheo = u0*exp(-lambda_*D*t);
    plot(x,u,'*',label='num')
    plot(x,utheo,'-',label='theo')
    xlim(*myax[:2])
    ylim(*myax[2:])
    xlabel(r'$x$');
    ylabel(r'$u(x,t)$');
    legend();
    title(f"Heat Equation RK4, t={t:0.2f}/{tmax-dt:0.2f}")
    draw()
    pause(1e-45)
    clf() if t != tmax-dt else show()

