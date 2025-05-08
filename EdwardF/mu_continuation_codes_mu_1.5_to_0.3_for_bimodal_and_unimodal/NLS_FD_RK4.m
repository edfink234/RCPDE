clear all;
close all;
g=-1;                % g = 1 is defocusing and g =-1 is focusing
N=401;               % number of mesh points
L=-10; R=10;         % Left and right bounds of interval
x=linspace(L,R,N)';  % discrete space
x=x(1:end-1);N=N-1;  % adjust for periodic BCs
dx=x(2)-x(1);        % mesh size
%Potential:
Omega=1*0.2;
b = 1;
Vp=0.5*Omega^2*x.^2;                % parabolic
H=1;wGaussian=0.5*1;
sech = @ (x) 1./cosh(x);
Vg= H * sech(b * x).^2;%H*exp(-x.^2./(2*wGaussian^2));   % Gaussian
V=Vp+Vg;

%Initial condition
A=H+0.2; c=0; x0=0*2.3;             % Amplitude, vel. & position
u0=A*sech(A*(x-x0)).*exp(1i*c*x);   % initial condition (IC)
u=u0; allu=u; t=0; allt=t; isave=1; % store solution and time

maxtime=20; dt=0.001;               % final time and time step
nsave=200; ndisp=20;                % snapshots to save/display
maxstep=round(maxtime/dt);
stopdisp=round(maxstep/ndisp);
stopsave=round(maxstep/nsave);
if(dt>sqrt(2)*dx^2/2) error('Probably need a smaller dt!'); end;

%Plot initial guess before Newton
% figure(1); clf;
t=0;
% plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
% axis([L R -1.1*A 1.1*A]);
% xlabel('$x$')
% ylabel('$|u(x,t)|$')
% title(['$t=',num2str(t),'$']);
% legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
fprintf('Press any key to continue...\n');pause

%Newton
fprintf('Let us do Newton iterations...\n');
rng('default');          % reset random generator
pert=0.0;                % size of perturbation
U=[real(u); imag(u)];  % initial guess for Newton
wfreq=1.5;             %temporal freq
itmax=15;
plot_newton=true;
newton_figure=1;
NLS_newton1D_complex
plot_newton=false;
neigs=50; % if neigs>0 only compute neigs evals
fprintf('Let us find stability...\n');
NLS_stability1D

wfreqs = wfreq-0.01:-.01:0.3;
alleig=zeros(neigs,length(wfreqs));
masses=zeros(length(wfreqs));
idx=0;

for wfreq = wfreqs
    idx = idx + 1;
    U=[real(u); imag(u)];  % initial guess for Newton, using the previous result for u
    up=u; %copy u into up
    NLS_newton1D_complex  % u is updated on the last line of Newton
    NewtonJump=abs(sum(abs(u)-abs(up)));
    if(NewtonJump>1 && idx>1)
        break;
    end
    
    NLS_stability1D
    alleig(:,idx)=ee; % store spectra
    masses(idx)=trapz(x, abs(u).^2);
end
%plot the masses for the wfreqs we just iterated over
figure(2);
plot(wfreqs, masses, '-');
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('$$M = \int_{-\infty}^{\infty} |u(x)|^2 \, dx$$', 'Interpreter', 'latex');
drawnow;

figure(3);
plot(wfreqs,imag(alleig),'.','MarkerSize',10);
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('Im(eigs)', 'Interpreter', 'latex');
title("Im(eigs) for bimodal u");
drawnow;

figure(4);
plot(wfreqs,real(alleig),'.','MarkerSize',10);
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('Real(eigs)', 'Interpreter', 'latex');
title("Re(eigs) for bimodal u");
drawnow;

A=H+2; c=0; x0=0*2.3;             % Amplitude, vel. & position
u0=A*sech(A*(x-x0)).*exp(1i*c*x);   % initial condition (IC)
u=u0; allu=u; t=0; allt=t; isave=1; % store solution and time
U=[real(u); imag(u)];  % initial guess for Newton
wfreq=1.5;
plot_newton=true;
newton_figure=5;
NLS_newton1D_complex
plot_newton=false;
neigs=50; % if neigs>0 only compute neigs evals
fprintf('Let us find stability...\n');
NLS_stability1D

alleig=zeros(neigs,length(wfreqs));
masses=zeros(length(wfreqs));
idx=0;

for wfreq = wfreqs
    idx = idx + 1;
    U=[real(u); imag(u)];  % initial guess for Newton, using the previous result for u
    up=u; %copy u into up
    NLS_newton1D_complex  % u is updated on the last line of Newton
    NewtonJump=abs(sum(abs(u)-abs(up)));
    if(NewtonJump>1 && idx>1)
        break;
    end
    
    NLS_stability1D
    alleig(:,idx)=ee; % store spectra
    masses(idx)=trapz(x, abs(u).^2);
end

figure(2);
hold on; % don't get rid of previous branch plz.
plot(wfreqs, masses, '-');
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('$$M = \int_{-\infty}^{\infty} |u(x)|^2 \, dx$$', 'Interpreter', 'latex');
legend('$\mu$ branch for bimodal $u$', '$\mu$ branch for unimodal $u$', 'Interpreter', 'latex');
title("Bifurcation diagram varying $\mu$ from 1.5 to 0.3 w/ $V_H = V_b = 1$", 'Interpreter', 'latex');
drawnow;

figure(6);
plot(wfreqs,imag(alleig),'.','MarkerSize',10);
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('Im(eigs)', 'Interpreter', 'latex');
title("Im(eigs) for unimodal u");
drawnow;

figure(7);
plot(wfreqs,real(alleig),'.','MarkerSize',10);
xlabel('$$\mu$$', 'Interpreter', 'latex');
ylabel('Real(eigs)', 'Interpreter', 'latex');
title("Re(eigs) for unimodal u");
drawnow;


%stop_here_ricardo



% return;
% u00=u;
% Maxu=max(abs(u));
%
% %displace IC:
% u=interp1(x,u,x-1e-8,'spline');u(isnan(u))=0; %Displace
%
% %Plot initial condition before integration
% figure(1);
% plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
% axis([L R -1.1*Maxu 1.1*Maxu]);
% xlabel('$x$')
% ylabel('$|u(x,t)|$')
% title(['$t=',num2str(t),'$']);
% legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
% fprintf('Press any key to continue...\n');pause
%
% %Integration
% for k=1:maxstep                     % main (time) loop
%  t=t+dt;
%  u = ODE_RK4(u,N,g,V,dx,dt);        % update using FD+RK4
%
%  if (round(k/stopsave)==k/stopsave) % Save progress of integration
%   isave=isave+1;
%   allt=[allt,t];
%   allu(:,isave) = u;
%  end
%  if (round(k/stopdisp)==k/stopdisp) % Plotting progress
%   figure(1);
%   plot(x,abs(u),x,real(u),'-',x,imag(u),'-',x,V,'-')
%   axis([L R -1.1*Maxu 1.1*Maxu]);
%   xlabel('$x$')
%   ylabel('$|u(x,t)|$')
%   title(['$t=',num2str(t),'$']);
%   legend('$|u|$','Re($u$)','Im($u$)','$V$'); drawnow;
%   %fprintf('Press any key to continue...\n');pause
%   drawnow
%  end
% end
