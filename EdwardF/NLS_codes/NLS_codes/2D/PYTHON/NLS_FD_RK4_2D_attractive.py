###################################################################
# NLS_FD_RK4.m:
# Integrate (finite differences in space and RK4 in time) the NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u 
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs, meshgrid, conj, array
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, draw, pause, show, close, legend, clf, subplots, tight_layout
from matplotlib.cm import coolwarm

from ODE_RK4_2D import ODE_RK4_2D

g=-1;                # g = 1 is defocusing and g =-1 is focusing
N=101;               # number of mesh points
L=-10; R=10;         # Left and right bounds of interval
x=linspace(L,R,N);  # discrete space
x = x[:-1];N=N-1;  # adjust for periodic BCs
y=x;
dx=x[1]-x[0];        # mesh size
X,Y = meshgrid(x,y)

A=2; c=1; x0=0;                     # Amplitude, vel. & position
sech = lambda x: 1/cosh(x)
U0=A*sech(A*(Y-x0))*exp(1j*c*Y);   # initial condition (IC)
#U0=A*sech(A*(X-x0))*exp(1j*c*X);   # initial condition (IC)
sx=1;sy=1;
#U0=A*exp(-(X**2/sx**2)-(Y**2/sy**2));   # initial condition (IC)

U=U0; allU=[U]; t=0; allt=[t]; isave=1; iMass=1; # store solution and time
# If no potential => V=0
try:
    V
except NameError:
    V=0;

maxtime=20; dt=0.001;               # final time and time step
nsave=200; nMass = 1000; ndisp=20;                # snapshots to save/display
maxstep=round(maxtime/dt);
stopdisp=round(maxstep/ndisp);
stopsave=round(maxstep/nsave);
stopMass=round(maxstep/nMass);
if(dt>sqrt(2)*dx**2/2):
    raise ValueError('Probably need a smaller dt!');

t=0;
Utheo=A*sech(A*(X-c*t-x0))*exp(1j*c*X+0.5j*(A**2-c**2)*t);
fig, ax = subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, abs(U), cmap = coolwarm)
#plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,abs(utheo),'k-')
#plot(x, abs(u), label = r'$|u|$')
#plot(x, real(u), '-', label = r'Re($u$)')
#plot(x, imag(u), '-', label = r'Im($u$)')
#plot(x, abs(utheo), 'k-', label = r'$|u_{\rm theo}|$')
xlim(L, R)
ylim(L, R)
ax.set_zlim(0, 2.2*A)
xlabel(r'$x$')
ylabel(r'$y$')
ax.set_zlabel(r'$|u(x,y,t)|$')

ax.set_title(f'$t={t:.2f}$')

#legend();

tight_layout()
pause(0.1)

Mass=sum(sum(U*conj(U)))*dx*dx;
allMass = [Mass]
alltMass=[0];
#exit(1)
# Main time loop
for k in range(1, maxstep + 1):
    t += dt
    
    U = ODE_RK4_2D(U,N,g,V,dx,dt,-A)
    Mass=sum(sum(U*conj(U)))*dx*dx;
    
    # Save progress of integration
    if round(k / stopMass) == k / stopMass:
        iMass += 1
        Mass=sum(sum(U*conj(U)))*dx*dx;
        alltMass.append(t)
        allMass.append(Mass)
    # Save progress of integration
    if round(k / stopsave) == k / stopsave:
        isave += 1
        allt.append(t)
        allU.append(U)
        
    # Plotting progress
    if round(k / stopdisp) == k / stopdisp:
        ax.cla()
#        utheo = A * sech(A * ((X - c * t - L) % (R - L) + L)) * exp(1j * c * X + 0.5j * (A**2 - c**2) * t)
        ax.plot_surface(X, Y, abs(U), cmap = coolwarm)
        xlim([L, R])
        ylim(L, R)
        ax.set_zlim(0, 2.2*A)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel(r'$|u(x,y,t)|$')
        ax.set_title(f'$t={t:.2f}$')
#        legend(['$|u|$', 'Re($u$)', 'Im($u$)', '$|u_{\mathrm{theo}}|$'])
        pause(.1)

show()  # Show final plot
Mass0 = allMass[0]
allMass = array(allMass)
plot(alltMass, (allMass-Mass0)/Mass0, '.-')
xlabel('$t$')
ylabel('$[M(t)-M(0)]/M(0)$')
show()
