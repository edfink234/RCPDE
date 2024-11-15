%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_FD_RK4.m:
% Integrate (finite differences in space and RK4 in time) the NLS: 
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u 
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g=-1;                % g = 1 is defocusing and g =-1 is focusing
N=401;               % number of mesh points
L=-10; R=10;         % Left and right bounds of interval
x=linspace(L,R,N)';  % discrete space
x=x(1:end-1);N=N-1;  % adjust for periodic BCs
dx=x(2)-x(1);        % mesh size

A=2; c=1; x0=0;                     % Amplitude, vel. & position 
u0=A*sech(A*(x-x0)).*exp(1i*c*x);   % initial condition (IC)
u=u0; allu=u; t=0; allt=t; isave=1; % store solution and time
if(exist('V')==0) V=0; end;         % If no potential => V=0

maxtime=20; dt=0.001;               % final time and time step
nsave=200; ndisp=20;                % snapshots to save/display
maxstep=round(maxtime/dt); 
stopdisp=round(maxstep/ndisp); 
stopsave=round(maxstep/nsave); 
if(dt>sqrt(2)*dx^2/2) error('Probably need a smaller dt!'); end; 

figure(1); clf; 
t=0;
utheo=A*sech(A*(x-c*t-x0)).*exp(1i*c*x+0.5i*(A^2-c^2)*t);
plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,abs(utheo),'k-')
axis([L R -1.1*A 1.1*A]); 
xlabel('$x$')
ylabel('$|u(x,t)|$')
title(['$t=',num2str(t),'$']);
legend('$|u|$','Re($u$)','Im($u$)','$|u_{\rm theo}|$'); drawnow;
fprintf('Press any key to continue...\n');pause

for k=1:maxstep                     % main (time) loop
 t=t+dt; 
 u = ODE_RK4(u,N,g,V,dx,dt);        % update using FD+RK4

 if (round(k/stopsave)==k/stopsave) % Save progress of integration
  isave=isave+1; 
  allt=[allt,t]; 
  allu(:,isave) = u; 
 end
 if (round(k/stopdisp)==k/stopdisp) % Plotting progress
  utheo=A*sech(A*(mod(x-c*t-x0-L,R-L)+L)).*exp(1i*c*x+0.5i*(A^2-c^2)*t);
  plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,abs(utheo),'k-')
  axis([L R -1.1*A 1.1*A]); 
  xlabel('$x$')
  ylabel('$|u(x,t)|$')
  title(['$t=',num2str(t),'$']);
  legend('$|u|$','Re($u$)','Im($u$)','$|u_{\rm theo}|$'); drawnow;
  %fprintf('Press any key to continue...\n');pause
  drawnow
 end
end
