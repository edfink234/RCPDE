%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ODE_RK4.m : RK4 integration
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function RK4 = ODE_RK4(u,N,g,V,dx,dt)
 k1 = dt*NLS_RHS(u       ,N,dx,g,V);
 k2 = dt*NLS_RHS(u+0.5*k1,N,dx,g,V);
 k3 = dt*NLS_RHS(u+0.5*k2,N,dx,g,V);
 k4 = dt*NLS_RHS(u+    k3,N,dx,g,V);
 RK4 = u + (k1 + 2*k2 + 2*k3 + k4)/6;
return;
