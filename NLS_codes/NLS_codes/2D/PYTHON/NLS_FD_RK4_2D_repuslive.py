###################################################################
# NLS_FD_RK4.m:
# Integrate (finite differences in space and RK4 in time) the NLS: 
#    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u 
# Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
# Code available at: 
#    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
###################################################################

from numpy import linspace, cosh, exp, sqrt, real, imag, abs, meshgrid, conj, arctan2 as atan2, tanh, angle, sin, exp, arcsin, cosh
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, title, draw, pause, show, close, legend, clf, subplots, tight_layout, figure, imshow, colorbar, gca
from matplotlib.cm import coolwarm

from ODE_RK4_2D import ODE_RK4_2D

g=+1;                # g = 1 is defocusing and g =-1 is focusing
N=201;               # number of mesh points
L=-10; R=10;         # Left and right bounds of interval
x=linspace(L,R,N);  # discrete space
x = x[:-1];N=N-1;  # adjust for periodic BCs
y=x;
dx=x[1]-x[0];        # mesh size
X,Y = meshgrid(x,y)
x1=-2;y1=0;
x2=+2;y2=0;
R1=sqrt((X-x1)**2+(Y-y1)**2);
R2=sqrt((X-x2)**2+(Y-y2)**2);
ANG1=atan2((Y-y1),(X-x1));
ANG2=atan2((Y-y2),(X-x2));

lw0=0.5;

mu=1; c=0; x0=0; S1=-1;S2=1; S=1;                    # AmplitUde, vel. & position
sech = lambda x: 1/cosh(x)

#U0=mu*tanh(mu*(X-x0))*exp(1j*c*Y);   # initial condition (IC)
#U0=mu*tanh(mu*RR)*exp(1j*S*ANG);   # initial condition (IC)

#U1=1*tanh(mu*R1)*exp(1j*S1*ANG1);
#U2=1*tanh(mu*R2)*exp(1j*S2*ANG2);
U1 = 1*((-(sin(R1)) / exp(R1)) / exp(arcsin(sech(R1))))*exp(1j*S1*ANG1);
U2 = 1*((-(sin(R2)) / exp(R2)) / exp(arcsin(sech(R2))))*exp(1j*S2*ANG1);
U0=mu*U1*U2;
sx=1;sy=1;
#U0=mu*exp(-(X**2/sx**2)-(Y**2/sy**2));   % initial condition (IC)

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
Utheo=mu*sech(mu*(X-c*t-x0))*exp(1j*c*X+0.5j*(mu**2-c**2)*t);

fig, ax = subplots(subplot_kw={"projection": "3d"})
abs_U = abs(U)**2
ax.plot_surface(X, Y, abs_U, cmap = coolwarm)
#plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,abs(utheo),'k-')
#plot(x, abs(u), label = r'$|u|$')
#plot(x, real(u), '-', label = r'Re($u$)')
#plot(x, imag(u), '-', label = r'Im($u$)')
#plot(x, abs(utheo), 'k-', label = r'$|u_{\rm theo}|$')
xlim(L, R)
ylim(L, R)
#ax.set_zlim(0, 1.1*mu)
ax.set_zlim(0, abs_U.max())
xlabel(r'$x$')
ylabel(r'$y$')
ax.set_zlabel(r'$|u(x,y,t)|$')

ax.set_title(f'density $t={t:.2f}$')

#legend();

tight_layout()
pause(0.1)
figure(12)
clf()
imshow(angle(U), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='auto')
xlabel('$x$')
ylabel('$y$')
title(f"phase t = ${t}$")
colorbar()
draw()
pause(.3)

Mass=sum(sum(U*conj(U)))*dx*dx;
allMass = [Mass]
alltMass=[0];
#exit()
# Main time loop
for k in range(1, maxstep + 1):
    t += dt
    
    U = ODE_RK4_2D(U,N,g,V,dx,dt,mu);
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
        figure(1)
        ax.cla()
#        utheo = A * sech(A * ((X - c * t - L) % (R - L) + L)) * exp(1j * c * X + 0.5j * (A**2 - c**2) * t)
        abs_U = abs(U)**2
        ax.plot_surface(X, Y, abs_U, cmap = coolwarm)
        xlim([L, R])
        ylim(L, R)
        ax.set_zlim(0, abs_U.max())
#        ax.set_zlim(0, 1.1*mu)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel(r'$|u(x,y,t)|$')
        ax.set_title(f'$t={t:.2f}$')
#        legend(['$|u|$', 'Re($u$)', 'Im($u$)', '$|u_{\mathrm{theo}}|$'])
        draw()
        
        
        figure(12)
        clf()
        imshow(angle(U), extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', aspect='auto')
        xlabel('$x$')
        ylabel('$y$')
        title(f"phase t = ${t:.2f}$")
        colorbar()
        draw()
        pause(.3)
        

show()  # Show final plot
plot(alltMass, allMass, '.-')
xlabel('$t$')
ylabel('$[M(t)-M(0)]/M(0)$')
show()
