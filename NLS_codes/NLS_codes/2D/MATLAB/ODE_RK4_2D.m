%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ODE_RK4_2D.m : RK4 integration
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function RK4 = ODE_RK4_2D(U,N,g,V,dx,dt)
%  k1 = dt*NLS_RHS_2D(U       ,N,dx,g,V);
%  k2 = dt*NLS_RHS_2D(U+0.5*k1,N,dx,g,V);
%  k3 = dt*NLS_RHS_2D(U+0.5*k2,N,dx,g,V);
%  k4 = dt*NLS_RHS_2D(U+    k3,N,dx,g,V);
%  RK4 = U + (k1 + 2*k2 + 2*k3 + k4)/6;
% return;

function RK4 = ODE_RK4_2D(U,N,g,V,dx,dt,mu)
 k1 = dt*NLS_RHS_2D(U       ,N,dx,g,V,mu);
 k2 = dt*NLS_RHS_2D(U+0.5*k1,N,dx,g,V,mu);
 k3 = dt*NLS_RHS_2D(U+0.5*k2,N,dx,g,V,mu);
 k4 = dt*NLS_RHS_2D(U+    k3,N,dx,g,V,mu);
 RK4 = U + (k1 + 2*k2 + 2*k3 + k4)/6;
return;