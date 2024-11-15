% Numerically integrate NLS in 2D:
% u_t = (i/2)*u_xx - i*g*(|u|^2)*u
% using finite differences in space and RK4 in time
% RCG Jul'24

D = 1;         % diffusion coefficient
Lx = -10;      % left end of domain in x
Rx = +10;      % right end of domain in x
Ly = -10;      % left end of domain in y
Ry = +10;      % right end of domain in y
Nx = 100;      % number of points in x direction
Ny = 100;      % number of points in y direction
dt=0.0001; % time step
tmax=10;  % max time
g=-1;      % non-linearity (g=-1: attractive; g=+1: repulsive)
dt0s=dt/6;

%Define discrete domain:
dx = (Rx - Lx) / Nx;
dy = (Ry - Ly) / Ny;
x=linspace(Lx,Rx-dx,Nx); % remove one dx to make solution periodic on [L,R]
y=linspace(Ly,Ry-dy,Ny); % remove one dx to make solution periodic on [L,R]
[X, Y] = meshgrid(x, y);

oodx2=1/dx^2;
oody2=1/dy^2;

%Define ICs:
A=1;          % amplitude for IC
v=1;          % initial vel
x0=0;         % initial pos
u0_x=A*sech(A*(X-x0)).*exp(1i*v*X);
maxu0_x=max(abs(u0_x));

%Define ICs:
A=1;          % amplitude for IC
v=1;          % initial vel
y0=0;         % initial pos
u0_y=A*sech(A*(Y-y0)).*exp(1i*v*Y);
maxu0_y=max(abs(u0_y));

u0 = u0_x .* u0_y; 
% u0 = A*sech(A*sqrt(X.^2+Y.^2)).*exp(1j*atan(Y./X));

minu0 = min(real(u0(:)));
maxu0 = max(real(u0(:)));
myax=[Lx Rx Ly Ry minu0 maxu0];

%nifty indexing for u_xx:
ip_x=[2:Nx,1];     % periodic BCs
im_x=[Nx,1:Nx-1];  % periodic BCs

%nifty indexing for u_yy:
ip_y=[2:Ny,1];     % periodic BCs
im_y=[Ny,1:Ny-1];  % periodic BCs

figure(1);clf
surf(X, Y, real(u0));
drawnow

u=u0;
ii=0;

for t=0:dt:tmax
 ii=ii+1;

 uxx1=oodx2*(u(:, im_x)-2*u+u(:, ip_x));
 uyy1=oody2*(u(:, im_y)-2*u+u(:, ip_y));
 k1 = 0.5j*(uxx1+uyy1) - 1j*g*(u.^2).*conj(u);

 u_temp = u+0.5*dt*k1;
 uxx2=oodx2*(u_temp(:, im_x)-2*u_temp+u_temp(:, ip_x));
 uyy2=oody2*(u_temp(:, im_y)-2*u_temp+u_temp(:, ip_y));
 k2 = 0.5j*(uxx2+uyy2) - 1j*g*(u_temp.^2).*conj(u_temp);
 
 u_temp = u+0.5*dt*k2;
 uxx3=oodx2*(u_temp(:, im_x)-2*u_temp+u_temp(:, ip_x));
 uyy3=oody2*(u_temp(:, im_y)-2*u_temp+u_temp(:, ip_y));
 k3 = 0.5j*(uxx3+uyy3) - 1j*g*(u_temp.^2).*conj(u_temp);

 u_temp = u+dt*k3;
 uxx4=oodx2*(u_temp(:, im_x)-2*u_temp+u_temp(:, ip_x));
 uyy4=oody2*(u_temp(:, im_y)-2*u_temp+u_temp(:, ip_y));
 k4 = 0.5j*(uxx4+uyy4) - 1j*g*(u_temp.^2).*conj(u_temp);
 
 u=u+(dt/6)*(k1+2*k2+2*k3+k4);

 if(mod(ii,1000)==1)
    subplot(2,1,1);
    % utheo = ...
    % surf(X, Y, real(utheo)); 
    % legend('theo');
    % title(['t=',num2str(t)])
    % axis(myax);
    % xlabel('x'); ylabel('y'); zlabel('u(x,y,t)');

    subplot(2,1,2);
    surf(X, Y, real(u)); 
    legend('num');
    title(['t=',num2str(t)])
    axis(myax);
    xlabel('x'); ylabel('y'); zlabel('u(x,y,t)');
    
    drawnow
 end
end
