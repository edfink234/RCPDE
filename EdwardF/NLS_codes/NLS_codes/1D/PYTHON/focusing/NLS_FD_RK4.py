###################################################################
# NLS_FD_RK4.m:
# Integrate (finite differences in space and RK4 in time) the NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u 
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, draw, pause, show, close, legend, clf
from ODE_RK4 import ODE_RK4

g=-1;                # g = 1 is defocusing and g =-1 is focusing
N=401;               # number of mesh points
L=-10; R=10;         # Left and right bounds of interval
x=linspace(L,R,N);  # discrete space
x = x[:-1];N=N-1;  # adjust for periodic BCs
dx=x[1]-x[0];        # mesh size

A=2; c=1; x0=0;                     # Amplitude, vel. & position
sech = lambda x: 1/cosh(x)
u0=A*sech(A*(x-x0))*exp(1j*c*x);   # initial condition (IC)
u=u0; allu=[u]; t=0; allt=[t]; isave=1; # store solution and time
# If no potential => V=0
try:
    V
except NameError:
    V=0;

maxtime=20; dt=0.001;               # final time and time step
nsave=200; ndisp=20;                # snapshots to save/display
maxstep=round(maxtime/dt);
stopdisp=round(maxstep/ndisp);
stopsave=round(maxstep/nsave);
if(dt>sqrt(2)*dx**2/2):
    raise ValueError('Probably need a smaller dt!');

t=0;
utheo=A*sech(A*(x-c*t-x0))*exp(1j*c*x+0.5j*(A**2-c**2)*t);

#plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,abs(utheo),'k-')
plot(x, abs(u), label = r'$|u|$')
plot(x, real(u), '-', label = r'Re($u$)')
plot(x, imag(u), '-', label = r'Im($u$)')
plot(x, abs(utheo), 'k-', label = r'$|u_{\rm theo}|$')
xlim(L, R)
ylim(-1.1*A, 1.1*A)
xlabel(r'$x$')
ylabel(r'$|u(x,t)|$')
title(rf'$t={t:.2f}$');
legend();
show()

# Main time loop
for k in range(1, maxstep + 1):
    t += dt
    
    u = ODE_RK4(u, N, dx, g, V, dt)
    
    # Save progress of integration
    if round(k / stopsave) == k / stopsave:
        isave += 1
        allt.append(t)
        allu.append(u)
        
    # Plotting progress
    if round(k / stopdisp) == k / stopdisp:
        utheo = A * sech(A * ((x - c * t - L) % (R - L) + L)) * exp(1j * c * x + 0.5j * (A**2 - c**2) * t)
        plot(x, abs(u), x, real(u), '-', x, imag(u), '-', x, abs(utheo), 'k-')
        xlim([L, R])
        ylim([-1.1 * A, 1.1 * A])
        xlabel('$x$')
        ylabel('$|u(x,t)|$')
        title(f'$t={t:.2f}$')
        legend(['$|u|$', 'Re($u$)', 'Im($u$)', '$|u_{\mathrm{theo}}|$'])
        draw()
        pause(0.3)  # Allows the plot to update
        clf()  # Clear current figure for the next plot

show()  # Show final plot


