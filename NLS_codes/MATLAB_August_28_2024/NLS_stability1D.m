%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_stability1D.m:
% Compute spectrum for a steady state for NLS: 
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 NLS_newton1D_complex;       % first find steady state usig Newton
 if(exist('V')==0) V=0; end; % If no potential just make it zero

 % Jacobian:
 M11 = -(-0.5*D2+diag(2*g*u.*conj(u)+V+w)); 
 M12 = -diag(g*u.*u); 
 M21 = -conj(M12); 
 M22 = -conj(M11); 
 M=1i*[M11,M12; M21,M22]; 

 neigs=0;                % if neigs>0 only compute neigs evals
 if(neigs>0)             % neigs evals about z0
  optionseigs.disp=0; 
  z0 = 0.5+0*1i;      
  [vvv,eee]=eigs(sparse(M),neigs,z0,optionseigs); 
 else                    % Full spectrum
  [vvv,eee]=eig(M); 
 end

ee=diag(eee);                          % eigenvalues
vv=vvv(1:Nx,:)+conj(vvv(Nx+1:end,:));  % eigenvectors
[~,bb]=(sort(real(ee))); 
bb=flipud(bb); ee=ee(bb); vv=vv(:,bb); % sort by largest real part

figure(2); clf
plot(ee,'o')
xlabel('Re(\lambda)')
ylabel('Im(\lambda)')
