%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_FD_RK4.m:
% Integrate (finite differences in space and RK4 in time) the NLS: 
%    i U_t = -(1/2) U_xx + g*|U|^2 U + V(x)*U 
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at: 
%    http://nonlinear.sdsU.edU/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

g=-1;                % g = 1 is defocUsing and g =-1 is focUsing
N=101;               % number of mesh points
L=-10; R=10;         % Left and right boUnds of interval
x=linspace(L,R,N);   % discrete space
x=x(1:end-1);N=N-1;  % adjUst for periodic BCs
y=x;                 % discrete space
dx=x(2)-x(1);        % mesh size
[X,Y]=meshgrid(x,y);

A=2; c=1; x0=0;                     % Amplitude, vel. & position 
U0=A*sech(A*(Y-x0)).*exp(1i*c*Y);   % initial condition (IC)
sx=1;sy=1;
%U0=A*exp(-(X.^2./sx^2)-(Y.^2./sy^2));   % initial condition (IC)
U=U0; allU(:,:,1)=U; t=0; allt=t; isave=1; iMass=1; % store solUtion and time
if(exist('V')==0) V=0; end;         % If no potential => V=0

maxtime=20; dt=0.001;               % final time and time step
nsave=200; nMass=1000; ndisp=20;                % snapshots to save/display
maxstep=round(maxtime/dt); 
stopdisp=round(maxstep/ndisp); 
stopsave=round(maxstep/nsave); 
stopMass=round(maxstep/nMass); 
if(dt>sqrt(2)*dx^2/2) error('Probably need a smaller dt!'); end; 

figure(1); clf; 
t=0;
Utheo=A*sech(A*(X-c*t-x0)).*exp(1i*c*X+0.5i*(A^2-c^2)*t);
surfl(X,Y,abs(U)); %,x,real(U),'-',x,imag(U),'-',x,abs(Utheo),'k-')
shading interp;colormap(gray(2048))
axis([L R L R -1.1*A 1.1*A]); 
axis tight
xlabel('$x$')
ylabel('$y$')
zlabel('$|U(,,y,t)|$')
title(['$t=',num2str(t),'$']);
%legend('$|U|$','Re($U$)','Im($U$)','$|U_{\rm theo}|$'); drawnow;
fprintf('Press any key to continue...');pause;fprintf('Running...\n')

Mass=sum(sum(U.*conj(U)))*dx*dx;
allMass(1)=Mass;
alltMass=0;

for k=1:maxstep                     % main (time) loop
 t=t+dt; 
 U = ODE_RK4_2D(U,N,g,V,dx,dt,-A);        % Update Using FD+RK4
 Mass=sum(sum(U.*conj(U)))*dx*dx;

 if (round(k/stopMass)==k/stopMass) % Save progress of integration
  iMass=iMass+1; 
  M=sum(sum(U.*conj(U)))*dx*dx;
  alltMass=[alltMass,t]; 
  allMass(iMass)=Mass;
 end
 if (round(k/stopsave)==k/stopsave) % Save progress of integration
  isave=isave+1; 
  allt=[allt,t]; 
  allU(:,:,isave) = U; 
 end
 if (round(k/stopdisp)==k/stopdisp) % Plotting progress
  Utheo=A*sech(A*(mod(X-c*t-x0-L,R-L)+L)).*exp(1i*c*X+0.5i*(A^2-c^2)*t);
  surfl(X,Y,abs(U)); %,x,real(U),'-',x,imag(U),'-',x,abs(Utheo),'k-')
  shading interp;colormap(gray(2048))
  axis([L R L R  -1.1*A 1.1*A]); 
  axis tight
  xlabel('$x$')
  ylabel('$y$')
  zlabel('$|U(,,y,t)|$')
  title(['$t=',num2str(t),'$']);
  %legend('$|U|$','Re($U$)','Im($U$)','$|U_{\rm theo}|$'); drawnow;
  %fprintf('Press any key to continUe...\n');pause
  drawnow
 end
end

%Plot conservation of mass:
figure(2)
Mass0=allMass(1);
plot(alltMass,(allMass-Mass0)./Mass0,'.-');
xlabel('$t$')
ylabel('$[M(t)-M(0)]/M(0)$')

