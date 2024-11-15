% Numerically integrate NLS in 1D:
% u_t = (i/2)*u_xx - i*g*(|u|^2)*u
% using finite differences in space and Euler in time
% RCG Jul'24

L=-10;   % left end of domain
R=+10;   % right end of domain
Nx=200;  % number of points in discrete domain
dt=0.0001; % time step
tmax=10;  % max time
g=-1;      % non-linearity (g=-1: attractive; g=+1: repulsive)

%Define discrete domain:
dx=(R-L)/Nx;
x=linspace(L,R-dx,Nx); % remove one dx to make solution periodic on [L,R]
oodx2=1/dx^2;

%Define ICs:
A=1;          % amplitude for IC
v=1;          % initial vel
x0=0;         % initial pos
u0=A*sech(A*(x-x0)).*exp(1i*v*x);
maxu0=max(abs(u0));
myax=[L R -1.2*maxu0 1.2*maxu0];

%nifty indexing for u_xx:
ip=[2:Nx,1];     % periodic BCs
im=[Nx,1:Nx-1];  % periodic BCs

figure(1);clf
plot(x,abs(u0),'*-')
drawnow

u=u0;
ii=0;

for t=0:dt:tmax
 ii=ii+1;

 uxx=oodx2*(u(im)-2*u+u(ip));

 
 RHS = 0.5j*uxx - 1j*g*(u.^2).*conj(u);
 u = u + dt*RHS; % Euler step
 if(mod(ii,1000)==1)
  utheo = A*sech(x-v*(t+dt)-x0).*exp(1i*v*x+0.5i*(A^2-v^2)*(t+dt));

  %plot(x,norm_u_squared,'*')%,x,utheo,'-')
  plot(x,real(u),'.',x,imag(u),'.',x,abs(u),'.')
  legend('real(u) num.','imag(u) num.', 'abs(u) num.');

  hold on
  plot(x,real(utheo),'-',x,imag(utheo),'-',x,abs(utheo),'-')
  legend('real(u) theory','imag theory', 'abs(u) theory');
  hold off
  axis(myax)

 
  xlabel('x'); ylabel('Re(u),Im(u),|u(x,t)|');
  title(['t=',num2str(t)]);
  drawnow
 end
end
