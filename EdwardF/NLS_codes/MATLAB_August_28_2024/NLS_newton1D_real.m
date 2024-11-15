%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_newton1D_real.m:
% Using Newton's method to compute a REAL steady state for NLS: 
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w=0.5;                 % temporal frequency of sought steady state
g=-1;                  % g = 1 is defocusing and g =-1 is focusing
L=-10;R=10;            % Left and right bounds of interval
Nx=301;                % number of mesh points
x=linspace(L,R,Nx)';   % discrete space
dx=x(2)-x(1);          % mesh size
ONE=ones(Nx,1);        % unit vector for the discrete Laplacian
D2=spdiags([ONE,-2*ONE,ONE],-1:1,Nx,Nx); % Discrete Laplacian 
D2(1,Nx)=1; D2(Nx,1)=1;                  % Periodic Boundary Conds
D2=D2/(dx^2);

A=sqrt(2*w);x0=0;       % Amplitude and position of initial guess
u0=A*sech(A*(x-x0));    % unperturbed initial guess (sech soliton)
rng('default');         % reset random generator
pert=1*0.01;            % size of perturbation
U=u0+pert*(rand(Nx,1)-0.5); % perturb IC = initial guess
it=0;err=1;             % initializing error

%Add potential
Omega=1*0.075;
V = Omega^2*x.^2/2 + 0.30*sech(x./2).^2;

while(err>1e-12)        % Main loop: checking Newton tolerance
 it=it+1;

 U2 = U.^2;
 J = -0.5*D2+diag(g*3*U2+V+w);   % Jacobian
 F = -0.5*D2*U+(g*U2+V+w).*U;    % RHS

 DU = -J\F;            % Newton correction
 U1 = U+DU;            % New step through Newton
 err=norm(F);          % How close to convergence we are

 figure(1)   % Plotting progress of iteration
 plot(x,U,'.',x,U1,x,V,'-',x,u0,'-'); 
 title(['it=',num2str(it),', error=',num2str(err)]);
 legend('previous','current','V(x)','initial','Location','NE')
 drawnow; fprintf('Press any key to continue...\n');pause

 U = U1;               % Update solution
end
u = U;

