%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_RHS.m : RHS for the NLS eqUation:
%    U_t = -i[-(1/2) (U_xx+U_yy) + g*|U|^2 U + V(x)*U]
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsU.edU/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function RHS = NLS_RHS_2D(U,N,dx,g,V)
% 
% %Remeber than inn Matlab: M(ydur,xdir,zdir,...)
% 
%  Upx = [U(:,2:N),U(:,1)];               % periodic BCs
%  Umx = [U(:,N),U(:,1:N-1)];             % periodic BCs
%  Upy = [U(2:N,:);U(1,:)];               % periodic BCs
%  Umy = [U(N,:);U(1:N-1,:)];             % periodic BCs
% 
%  % Upx = [U(:,2:N),U(:,N-1)];             % Neumann BCs
%  % Umx = [U(:,2),U(:,1:N-1)];             % Neumann BCs
%  % Upy = [U(2:N,:);U(N-1,:)];             % Neumann BCs
%  % Umy = [U(2,:);U(1:N-1,:)];             % Neumann BCs
% 
% % Up = [U(2:N);0];                  % Zero (Dirichlet) BCs
% % Um = [0;U(1:N-1)];                % Zero (Dirichlet) BCs
% 
% % Up = [U(2:N);U(N)-(U(N-1)-U(N))]; % linear extrapolation BCs
% % Um = [U(1)-(U(2)-U(1));U(1:N-1)]; % linear extrapolation BCs
% 
% % Up = [U(2:N);U(N)];               % Laplacian Zero BCs
% % Um = [U(1);U(1:N-1)];             % Laplacian Zero BCs
% 
%   Uxx = (0.5/dx^2)*(Upx-2*U+Umx);
%   Uyy = (0.5/dx^2)*(Upy-2*U+Umy);
% 
%   RHS = 1i*(Uxx + Uyy - (g*U.*conj(U)+V).*U);
% 
% return;

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
