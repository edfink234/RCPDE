%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_FD_RK4_2D.m:
% Integrate (finite differences in space and RK4 in time) the 2D NLS:
%    i u_t = -(1/2) (u_xx + u_yy) + g*|u|^2 u + V(x)*u
% Adapted from 1D code by 
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function RHS = NLS_RHS_2D(u, Nx, Ny, dx, dy, g, V)

    % Reshape for 2D operations
    % disp(size(u)); disp(Nx); disp(Ny);
    u = reshape(u, Nx, Ny);
    V = reshape(V, Nx, Ny);

    % Periodic BCs (adjust if needed)
    up_x = [u(Nx,:); u(1:Nx-1,:)];
    um_x = [u(2:Nx,:); u(1,:)];
    up_y = [u(:,Ny), u(:,1:Ny-1)];
    um_y = [u(:,2:Ny), u(:,1)];

    % 2D Laplacian
    Laplacian_u = (0.5/dx^2)*(up_x - 2*u + um_x) + ...
                  (0.5/dy^2)*(up_y - 2*u + um_y);

    RHS = 1i*(Laplacian_u - (g*u.*conj(u) + V).*u);

end

function RK4 = ODE_RK4_2D(u, Nx, Ny, g, V, dx, dy, dt)
    k1 = dt*NLS_RHS_2D(u, Nx, Ny, dx, dy, g, V);
    % disp("k1 size: "); disp(size(k1));
    k2 = dt*NLS_RHS_2D(u + 0.5*k1, Nx, Ny, dx, dy, g, V);
    k3 = dt*NLS_RHS_2D(u + 0.5*k2, Nx, Ny, dx, dy, g, V);
    k4 = dt*NLS_RHS_2D(u + k3, Nx, Ny, dx, dy, g, V);
    RK4 = u + (k1 + 2*k2 + 2*k3 + k4)/6;
end

g=-1;                % g = 1 is defocusing and g =-1 is focusing
N=201;               % number of mesh points in each direction
L=-10; R=10;         % Left and right bounds of interval
x=linspace(L,R,N)';  % discrete space in x
y=x;                 % discrete space in y (assuming square domain)
x=x(1:end-1);N=N-1;  % adjust for periodic BCs
y=y(1:end-1); 
dx=x(2)-x(1);        % mesh size in x
dy=y(2)-y(1);        % mesh size in y

[X,Y] = meshgrid(x,y);
[Nx, Ny] = size(X);

% Parameters for initial condition
x1=0; y1=0; 
x2=3; y2=0;
mu=1; S1=-1; S2=1;

R1=sqrt((X-x1).^2+(Y-y1).^2);
R2=sqrt((X-x2).^2+(Y-y2).^2);
ANG1=atan2((Y-y1),(X-x1));
ANG2=atan2((Y-y2),(X-x2));

U1=1*tanh(mu*R1).*exp(1i*S1*ANG1);   % IC: approx vortex 1
U2=1*tanh(mu*R2).*exp(1i*S2*ANG2);   % IC: approx vortex 2
U0=mu*U1.*U2;
u=U0;

%Potential:
Omega=1*0.2;
Vp=0.5*Omega^2*(X.^2 + Y.^2);        % 2D parabolic potential

H=1*0.5; w=1; 
Vg=H*exp(-(X.^2 + Y.^2)./(2*w^2));    % 2D Gaussian

V=Vp+Vg;

maxtime=40; dt=0.001;         % final time and time step
nsave=200; ndisp=200;          % snapshots to save/display
maxstep=round(maxtime/dt);
stopdisp=round(maxstep/ndisp);
stopsave=round(maxstep/nsave);
if(dt>sqrt(2)*dx^2/2) 
    error('Probably need a smaller dt!'); 
end;

% Discrete Laplacian in 2D (same as before)
ONE_x = ones(Nx,1);
ONE_y = ones(Ny,1);
D2x = spdiags([ONE_x,-2*ONE_x,ONE_x],-1:1,Nx,Nx); 
D2x(1,Nx) = 1; D2x(Nx,1) = 1; 
D2x = D2x/(dx^2);

D2y = spdiags([ONE_y,-2*ONE_y,ONE_y],-1:1,Ny,Ny); 
D2y(1,Ny) = 1; D2y(Ny,1) = 1;
D2y = D2y/(dy^2);

D2 = kron(speye(Ny),D2x) + kron(D2y,speye(Nx));

% Indices for real & imaginary parts
indR = 1:Nx*Ny; 
indI = Nx*Ny+1:2*Nx*Ny; 

% Reshape for Newton's method
U = [real(u(:)); imag(u(:))];
V = V(:);

% Find steady state using Newton's method
w = max(max(abs(u)))^2/2;  % Estimate temporal frequency
itmax = 15;

% NLS_newton2D_complex 
% ====================
if(exist('itmax')==0) itmax=10;end % Number of iterations

% Define Spatial Grid 
[X,Y] = meshgrid(x,y);
[Nx, Ny] = size(X);

% Discrete Laplacian in 2D
ONE_x = ones(Nx,1);
ONE_y = ones(Ny,1);
D2x = spdiags([ONE_x,-2*ONE_x,ONE_x],-1:1,Nx,Nx); % Laplacian in x
D2x(1,Nx) = 1; D2x(Nx,1) = 1; % Periodic BCs in x
D2x = D2x/(dx^2);

D2y = spdiags([ONE_y,-2*ONE_y,ONE_y],-1:1,Ny,Ny); % Laplacian in y
D2y(1,Ny) = 1; D2y(Ny,1) = 1; % Periodic BCs in y
D2y = D2y/(dy^2);

% Create 2D Laplacian using Kronecker product
D2 = kron(speye(Ny),D2x) + kron(D2y,speye(Nx));

% Indices for real & imaginary parts
indR = 1:Nx*Ny; 
indI = Nx*Ny+1:2*Nx*Ny; 

it=0; err=1; % initializing error

if(exist('V')==0) 
    V = zeros(Nx, Ny); 
end; % If no potential => V=0

% Reshape V and U for 2D
V = V(:);
U = U(:);

while((err>1e-10)&(it<itmax)) % Main loop
    it=it+1;
    Ur=U(indR); Ui=U(indI); % real and imag parts of u

    U2 = Ur.^2 + Ui.^2; % mod square of u

    % Jacobian in 2D
    J11 = -0.5*D2 + spdiags(g*(3*Ur.^2 + Ui.^2) + V + w, 0, Nx*Ny, Nx*Ny); 
    J22 = -0.5*D2 + spdiags(g*(Ur.^2 + 3*Ui.^2) + V + w, 0, Nx*Ny, Nx*Ny); 
    J12 = spdiags(g*2*Ur.*Ui, 0, Nx*Ny, Nx*Ny); 
    J = [J11, J12; J12, J22]; 

    % RHS in 2D
    Fr = -0.5*D2*Ur + (g*U2 + V + w).*Ur; 
    Fi = -0.5*D2*Ui + (g*U2 + V + w).*Ui;
    F = [Fr; Fi];

    DU = -J\F; % Newton correction
    U = U + DU; 
    disp("Size of F = ");disp(size(F));
    err = norm(F); 

    % Plotting (adapt for 2D visualization)
    figure(1)
    subplot(1,2,1)
    imagesc(x,y,reshape(U(indR),Nx,Ny)); colorbar; title('Re(U)')
    subplot(1,2,2)
    imagesc(x,y,reshape(U(indI),Nx,Ny)); colorbar; title('Im(U)')
    drawnow; 
    fprintf('Press any key to continue...\n'); 
    pause
    disp(err);
end

u = reshape(U(indR) + 1i*U(indI), Nx, Ny);

disp(size(u))


% Slightly displace the solution (optional)
u = interp2(x,y,u,x-0.01,y-0.01,'spline');
u(isnan(u)) = 0;
Maxu = max(max(abs(u)));

% Plotting initial condition
figure(1); clf;
t = 0;
subplot(1,2,1)
imagesc(x,y,abs(u)); colorbar; title('Initial condition: |u|')
subplot(1,2,2)
imagesc(x,y,angle(u)); colorbar; title('Initial condition: arg(u)')
allu(:,:,1)=u; t=0; allt=t; isave=1;

% Main time loop
for k=1:maxstep
    % fprintf("k = %d\n",k);
    t = t + dt;
    u = ODE_RK4_2D(u, Nx, Ny, g, V, dx, dy, dt); 

    if (round(k/stopsave)==k/stopsave) 
        isave = isave + 1;
        allt = [allt, t];
        allu(:,:,isave) = u;
    end

    if (round(k/stopdisp)==k/stopdisp) 
        subplot(1,2,1)
        imagesc(x,y,abs(u)); colorbar; title(['|u| at t = ', num2str(t)])
        subplot(1,2,2)
        imagesc(x,y,angle(u)); colorbar; title(['arg(u) at t = ', num2str(t)])
        drawnow;
    end
end