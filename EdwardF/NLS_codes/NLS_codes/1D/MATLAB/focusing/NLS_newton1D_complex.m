%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_newton1D_complex.m:
% Using Newton's method to compute a COMPLEX steady state for NLS: 
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w =0.5;                 % temporal frequency of sought steady state
g =-1;                  % g = 1 is defocusing and g =-1 is focusing
L =-10; R=10;           % Left and right bounds of interval
Nx=301;                 % number of mesh points
x =linspace(L,R,Nx)';   % discrete space
dx=x(2)-x(1);           % mesh size
ONE=ones(Nx,1);         % unit vector for the discrete Laplacian
D2=spdiags([ONE,-2*ONE,ONE],-1:1,Nx,Nx); % Discrete Laplacian 
D2(1,Nx)=1; D2(Nx,1)=1;                  % Periodic Boundary Conds
D2=D2/(dx^2); 
indR=1:Nx; indI=Nx+1:2*Nx;               % index for real & imag

A=sqrt(2*w);x0=(R+L)/2;  % Amplitude and position of initial guess
u0=A*sech(A*(x-x0));     % unperturbed initial guess (sech soliton)
rng('default');          % reset random generator
pert=0.3;                % size of perturbation
up=u0+pert*(rand(Nx,1)-0.5).*exp(-x.^2/10); % perturb IC
U=[real(up); imag(up)];  % initial guess
it=0; err=1;             % initializing error

if(exist('V')==0) V=0; end;  % If no potential => V=0

while(err>1e-10)             % Main loop: checking Newton tolerance
 it=it+1; 
 Ur=U(indR); Ui=U(indI);     % real and imag parts of u

 U2 = Ur.^2+Ui.^2;                          % mod square of u
 J11=-0.5*D2+diag(g*(3*Ur.^2+  Ui.^2)+V+w); % J11 part of Jacobian
 J22=-0.5*D2+diag(g*(  Ur.^2+3*Ui.^2)+V+w); % J22 part of Jacobian
 J12=g*diag(2*Ur.*Ui);                      % J12 part of Jacobian
 J = [J11,J12; J12,J22];                    % Full Jacobian
 Fr = -0.5*D2*Ur+(g*U2+V+w).*Ur;            % real(RHS)
 Fi = -0.5*D2*Ui+(g*U2+V+w).*Ui;            % imag(RHS)
 F = [Fr; Fi];                              % RHS

 DU = -J\F;                  % Newton correction
 U1 = U+DU;                  % New step through Newton
 err=norm(F);                % How close to convergence we are

 figure(1)                   % Plotting progress of iteration
 plot(x,U(indR),'.',x,U(indI),'.',x,U1(indR),x,U1(indI),x,V)
 xlabel('x'); ylabel('u'); 
 title(['it=',num2str(it),', error=',num2str(err)]); 
 legend('Re(previous)','Im(previous)','Re(current)',...
        'Im(current)','V(x)','Location','NE')
 drawnow; fprintf('Press any key to continue...\n'); pause 

 U = U1;                     % Update solution
end
u = U(indR)+1i*U(indI);      % wrapping into a complex vector
