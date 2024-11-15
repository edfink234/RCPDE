%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_RHS.m : RHS for the NLS equation:
%    u_t = -i[-(1/2) u_xx + g*|u|^2 u + V(x)*u]
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RHS = NLS_RHS(u,N,dx,g,V)

  up = [u(N);u(1:N-1)];             % periodic BCs
  um = [u(2:N);u(1)];               % periodic BCs

% up = [0;u(1:N-1)];                % Zero BCs
% um = [u(2:N);0];                  % Zero BCs

% up = [u(1)-(u(2)-u(1));u(1:N-1)]; % linear extrapolation BCs
% um = [u(2:N);u(N)-(u(N-1)-u(N))]; % linear extrapolation BCs

% up = [u(1);u(1:N-1)];             % Laplacian Zero BCs
% um = [u(2:N);u(N)];               % Laplacian Zero BCs

  RHS = 1i*((0.5/dx^2)*(up-2*u+um) - (g*u.*conj(u)+V).*u);

return;
