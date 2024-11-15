% Numerically integrate NLS in 1D:
% u_t = (i/2)*u_xx - i*g*(|u|^2)*u
% using finite differences in space and RK4 in time
% RCG Jul'24

L=-10;     % left end of domain
R=+10;     % right end of domain
Nx=200;    % number of points in discrete domain
dt=0.0001;   % time step
tmax=10;   % max time
g=-1;      % non-linearity (g=-1: attractive; g=+1: repulsive)

%Define discrete domain:
dx=(R-L)/Nx;
x=linspace(L,R-dx,Nx); % remove one dx to make solution periodic on [L,R]
oodx2=1/dx^2;
dt0s=dt/6;

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
%ip=[2:Nx,Nx];     % zero Lap BCs
%im=[1,1:Nx-1];  % zero Lap BCs

figure(1);clf
plot(x,abs(u0),'*-')
drawnow

u=u0;
ii=0;
for t=0:dt:tmax
 ii=ii+1;

 uxx1=oodx2*(u(im)-2*u+u(ip));
 k1 = 0.5j*uxx1 - 1j*g*(u.^2).*conj(u);

 u_temp = u+0.5*dt*k1;
 uxx2=oodx2*(u_temp(im)-2*u_temp+u_temp(ip));
 k2 = 0.5j*uxx2 - 1j*g*(u_temp.^2).*conj(u_temp);

 u_temp = u+0.5*dt*k2;
 uxx3=oodx2*(u_temp(im)-2*u_temp+u_temp(ip));
 k3 = 0.5j*uxx3 - 1j*g*(u_temp.^2).*conj(u_temp);

 u_temp = u+dt*k3;
 uxx4=oodx2*(u_temp(im)-2*u_temp+u_temp(ip));
 k4 = 0.5j*uxx4 - 1j*g*(u_temp.^2).*conj(u_temp);

 u=u+dt0s*(k1+2*k2+2*k3+k4);
 
 %norm_u_squared = abs(u).^2;

 if(mod(ii,1000)==1)
  utheo = A*sech(x-v*(t+dt)-x0).*exp(1i*v*x+0.5i*(A^2-v^2)*(t+dt));

  %plot(x,norm_u_squared,'*')%,x,utheo,'-')
  plot(x,real(u),'.',x,imag(u),'.',x,abs(u),'.')
  legend('real','imag', 'abs(u)');

  hold on
  plot(x,real(utheo),'-',x,imag(utheo),'-',x,abs(utheo),'-')
  legend('real','imag', 'abs(u)');
  hold off
  axis(myax)

 
  xlabel('x'); ylabel('Re(u),Im(u),|u(x,t)|');
  title(['t=',num2str(t)]);
  drawnow
 end
end
