%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_newton1D_complex.m:
% Using Newton's method to compute a COMPLEX steady state for NLS: 
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(exist('itmax')==0) itmax=20;end


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
 err=norm(F)             % How close to convergence we are

 figure(1)                   % Plotting progress of iteration
 plot(x,U(indR),'.',x,U(indI),'.',x,U1(indR),x,U1(indI),x,V)
 xlabel('x'); ylabel('u'); 
 title(['it=',num2str(it),', error=',num2str(err)]); 
 legend('Re(previous)','Im(previous)','Re(current)',...
        'Im(current)','V(x)','Location','NE')
 drawnow; fprintf('Press any key to continue...\n');

 U = U1;                     % Update solution
end
u = U(indR)+1i*U(indI);      % wrapping into a complex vector
