% Numerically integrate heat equation in 2D:
% u_t = D (u_xx + u_yy)
% using finite differences in space and Euler in time
% RCG Jul'24

D = 1;         % diffusion coefficient
Lx = -10;      % left end of domain in x
Rx = +10;      % right end of domain in x
Ly = -10;      % left end of domain in y
Ry = +10;      % right end of domain in y
Nx = 100;      % number of points in x direction
Ny = 100;      % number of points in y direction
dt = 0.01;     % time step
tmax = 3;      % max time

%Define discrete domain:
dx = (Rx - Lx) / Nx;
dy = (Ry - Ly) / Ny;
x=linspace(Lx,Rx-dx,Nx); % remove one dx to make solution periodic on [L,R]
y=linspace(Ly,Ry-dy,Ny); % remove one dx to make solution periodic on [L,R]
[X, Y] = meshgrid(x, y);

oodx2=1/dx^2;
oody2=1/dy^2;

%Define ICs:
N0_x=3;         %number of periods in x
WL_x=(Rx-Lx)/N0_x;  % wavelength of IC in x
k0_x=2*pi/WL_x;   % wavenumber of IC
lambda_x=k0_x^2;  % exp decay rate for this IC
u0_x=sin(k0_x*X); % IC for x

N0_y=3;         %number of periods in y
WL_y=(Ry-Ly)/N0_y;  % wavelength of IC in y
k0_y=2*pi/WL_y;   % wavenumber of IC
lambda_y=k0_y^2;  % exp decay rate for this IC
u0_y=sin(k0_y*Y); % IC for y

u0 = u0_x .* u0_y;

minu0 = min(u0(:));
maxu0 = max(u0(:));
myax=[Lx Rx Ly Ry minu0 maxu0];

%nifty indexing for u_xx:
ip_x=[2:Nx,1];     % periodic BCs
im_x=[Nx,1:Nx-1];  % periodic BCs

%nifty indexing for u_yy:
ip_y=[2:Ny,1];     % periodic BCs
im_y=[Ny,1:Ny-1];  % periodic BCs

figure(1);clf
surf(X, Y, u0);
drawnow

u=u0;
for t=0:dt:tmax

 uxx=oodx2*(u(:, im_x)-2*u+u(:, ip_x));
 uyy=oody2*(u(:, im_y)-2*u+u(:, ip_y));
 RHS = D * (uxx + uyy); %+g*u.^2.*conj(u);
 u = u + dt*RHS; % Euler step

 subplot(2,1,1);
 utheo = u0*exp(-(lambda_x + lambda_y)*D*t);
 surf(X, Y, utheo); 
 legend('theo');
 title(['t=',num2str(t)])
 axis(myax);
 xlabel('x'); ylabel('y'); zlabel('u(x,y,t)');

 subplot(2,1,2);
 surf(X, Y, u); 
 legend('num');
 title(['t=',num2str(t)])
 axis(myax);
 xlabel('x'); ylabel('y'); zlabel('u(x,y,t)');

 drawnow
end
