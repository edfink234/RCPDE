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

A=2; c=0; x0=0*2.3;                 % Amplitude, vel. & position
u0=A*sech(A*(x-x0)).*exp(1i*c*x);   % initial condition (IC)
u=u0; allu=u; t=0; allt=t; isave=1; % store solution and time

%Potential:
Omega=1*0.2;
Vp=0.5*Omega^2*x.^2;                     % parabolic
H=1*0.5;w=1;Vg=H*exp(-x.^2./(2*w^2));    % Gaussian
V=Vp+Vg;

maxtime=40; dt=0.001;               % final time and time step
nsave=200; ndisp=20;                % snapshots to save/display
maxstep=round(maxtime/dt);
stopdisp=round(maxstep/ndisp);
stopsave=round(maxstep/nsave);
if(dt>sqrt(2)*dx^2/2) error('Probably need a smaller dt!'); end;

figure(1); clf;
t=0;
plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
axis([L R -1.1*A 1.1*A]);
xlabel('$x$')
ylabel('$|u(x,t)|$')
title(['$t=',num2str(t),'$']);
legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
fprintf('Press any key to continue...\n');pause

fprintf('Let us do Newton iterations...\n');
rng('default');          % reset random generator
pert=0.0;                % size of perturbation
up=u0+pert*(rand(N,1)-0.5)+1i*pert*(rand(N,1)-0.5); % perturb IC
U=[real(up); imag(up)];  % initial guess for Newton
w=A^2/2;                 %temporal freq
itmax=15;
NLS_newton1D_complex
u=interp1(x,u,x-0.01,'spline');u(isnan(u))=0; %Displace
u00=u;
Maxu=max(abs(u));

figure(1);
plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
axis([L R -1.1*Maxu 1.1*Maxu]);
xlabel('$x$')
ylabel('$|u(x,t)|$')
title(['$t=',num2str(t),'$']);
legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
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
  plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
  axis([L R -1.1*Maxu 1.1*Maxu]);
  xlabel('$x$')
  ylabel('$|u(x,t)|$')
  title(['$t=',num2str(t),'$']);
  legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
  %fprintf('Press any key to continue...\n');pause
  drawnow
 end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_newton1D_complex.m:
% Using Newton's method to compute a COMPLEX steady state for NLS:
%    U_t = -i[-(1/2) (U_xx+U_yy) + g*|U|^2 U + V(x)*U]
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(exist('itmax')==0) itmax=10;end

ONE=ones(N,1);         % unit vector for the discrete Laplacian
D2=spdiags([ONE,-2*ONE,ONE],-1:1,N,N); % Discrete Laplacian
D2(1,N)=1; D2(N,1)=1;                  % Periodic Boundary Conds
D2=D2/(dx^2);
indR=1:N; indI=N+1:2*N;               % index for real & imag

it=0; err=1;             % initializing error

if(exist('V')==0) V=0; end;  % If no potential => V=0

while((err>1e-10)&(it<itmax)) % Main loop: checking Newton tolerance
 it=it+1
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ODE_RK4.m : RK4 integration
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RK4 = ODE_RK4_2D(U,N,g,V,dx,dt,mu)
 k1 = dt*NLS_RHS_2D(U       ,N,dx,g,V,mu);
 k2 = dt*NLS_RHS_2D(U+0.5*k1,N,dx,g,V,mu);
 k3 = dt*NLS_RHS_2D(U+0.5*k2,N,dx,g,V,mu);
 k4 = dt*NLS_RHS_2D(U+    k3,N,dx,g,V,mu);
 RK4 = U + (k1 + 2*k2 + 2*k3 + k4)/6;
return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_RHS.m : RHS for the NLS eqUation:
%    U_t = -i[-(1/2) (U_xx+U_yy) + g*|U|^2 U + V(x)*U]
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsU.edU/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RHS = NLS_RHS(U,N,dx,g,V,mu)

%Remeber than inn Matlab: M(ydur,xdir,zdir,...)

% Upx = [U(:,2:N),U(:,1)];               % periodic BCs
% Umx = [U(:,N),U(:,1:N-1)];             % periodic BCs
% Upy = [U(2:N,:);U(1,:)];               % periodic BCs
% Umy = [U(N,:);U(1:N-1,:)];             % periodic BCs

 Upx = [U(:,2:N),U(:,N-1)];             % Neumann BCs
 Umx = [U(:,2),U(:,1:N-1)];             % Neumann BCs
 Upy = [U(2:N,:);U(N-1,:)];             % Neumann BCs
 Umy = [U(2,:);U(1:N-1,:)];             % Neumann BCs

% Up = [U(2:N);0];                  % Zero (Dirichlet) BCs
% Um = [0;U(1:N-1)];                % Zero (Dirichlet) BCs

% Up = [U(2:N);U(N)-(U(N-1)-U(N))]; % linear extrapolation BCs
% Um = [U(1)-(U(2)-U(1));U(1:N-1)]; % linear extrapolation BCs

% Up = [U(2:N);U(N)];               % Laplacian Zero BCs
% Um = [U(1);U(1:N-1)];             % Laplacian Zero BCs

 Uxx = (0.5/dx^2)*(Upx-2*U+Umx);
 Uyy = (0.5/dx^2)*(Upy-2*U+Umy);

 RHS = 1i*(Uxx + Uyy - (g*U.*conj(U)+V-mu).*U);

return;

