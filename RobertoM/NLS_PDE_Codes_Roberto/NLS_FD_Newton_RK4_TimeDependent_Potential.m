%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_FD_Newton_RK4_TDP.m:
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
u=u0; t=0; allt=t; isave=1;isaveCM = 1;alltCM = t;% store solution and time
isavemass = 1;
idisp = 1;

A0=1;nu=0;B0=0.0;gamma=0;phi1=0;C0=2;omega=5;phi2=pi/2; %RCG
params = [A0,B0,C0,nu,gamma,omega,phi1,phi2]';

Omega1_est= @(C0,omega)(-0.6520./C0 + .4599./omega ...
    -.7141./(C0.*omega) + 6.0418./C0.^2 -1.2192./omega.^2 -7.3747./...
    C0.^3+.9255./omega.^3 -.4726./(omega.*C0.^2)+1.4557./...
    (C0.*omega.^2));

omega1_est = abs(Omega1_est(C0,omega));


T = ceil(2*pi/omega1_est);
% T = 100;




maxtime=10*T; dt=0.001;               % final time and time step
nsave=200; ndisp=10000;   % snapshots to save/display
nsaveCM = 1000; %Snapshots to compute and save center of mass
maxstep=round(maxtime/dt); 
stopdisp=round(maxstep/ndisp); 
stopsave=round(maxstep/nsave); 
stopsaveCM = round(maxstep/nsaveCM);
if(dt>sqrt(2)*dx^2/2);error('Probably need a smaller dt!'); end

t0 = 0;
t=t0;
V = TDPotential(x,t,params);

fprintf('Let us do Newton iterations...\n');
rng('default');          % reset random generator
pert=0.0;                % size of perturbation
up=u0+pert*(rand(N,1)-0.5)+1i*pert*(rand(N,1)-0.5); % perturb IC
U=[real(up); imag(up)];  % initial guess for Newton
w=A^2/2;                 %temporal freq
itmax=15;
NLS_newton1D_complex
G = 0;%Perturbation scale factor 
u=interp1(x,u,x-G*0.001,'spline');u(isnan(u))=0; %Displace
u00=u;
Maxu=max(abs(u));
Maxmaxu = max(max(abs(alluCM)));
allu=u;alluCM = u;

density = u.*conj(u);
xmax = sum(x.*density)*dx/(sum(density)*dx);
xpeak_vals = xmax;

flag = 0;%Break condition
for k=1:maxstep                     % main (time) loop
 t=t+dt; 
 V = TDPotential(x,t,params);
 u = ODE_RK4(u,N,g,V,dx,dt);        % update using FD+RK4
 if (round(k/stopsaveCM)==k/stopsaveCM)%Save center of mass progress
     % and mass conservation
     isaveCM = isaveCM + 1;
     alltCM = [alltCM,t];
     alluCM(:,isaveCM) = u;
     density = u.*conj(u);%Computing density
     xmax = sum(x.*density)*dx/(sum(density)*dx);%Finding the CM
     xpeak_vals(isaveCM) = xmax;%Storing the location

     if xmax > abs(C0)
         flag = 1;
     end
 end
 if flag
     fprintf("Soliton motion is not periodic...\n");
     break 
 end
 if (round(k/stopsave)==k/stopsave) % Save progress of integration
  isave=isave+1; 
  allt=[allt,t]; 
  allu(:,isave) = u; 
 end
 % if (round(k/stopdisp)==k/stopdisp) % Plotting progress
 %   idisp = idisp +1;
 %  plot(x,abs(u),'k-',x,V,'-')
 %    % hold on 
 %    % plot([-0.05,-0.05],[-1.1*Maxu,1.1*Maxu],'k--')
 %    % plot([0.05,0.05],[-1.1*Maxu,1.1*Maxu],'k--')
 %    % hold off
 %  axis([L R -0.1*Maxu 1.1*Maxu]); 
 %  xlabel('$x$')
 %  ylabel('$|u(x,t)|$')
 %  title(['$t=',num2str(t),'$']);
 %  legend('$|u|$','$V$'); drawnow;
 %  %fprintf('Press any key to continue...\n');pause
 %  drawnow
 %   Q(idisp) = getframe;
 % end
end


% writerObj = VideoWriter('Soliton_Motion_Linear_ControlGroup','MPEG-4');
% open(writerObj);writeVideo(writerObj,Q); close(writerObj);


if ~flag

%Fast Fourier Transform
NLS_FFT
% Ts = alltCM(2)-alltCM(1);
% fs = 1/Ts;
% y = fft(xpeak_vals);
% ly = length(y);
% y_oneside = y(1:floor(ly/2));
% f = fs*(0:ly/2-1)/ly;
% y_meg = abs(y_oneside)/(floor(ly/2));
% duration = maxtime;
% figure(3)
% stem(f,y_meg)
% xlabel("Frequency(Hz)");ylabel("Amplitude")
a1 = pk1;a2 = pk2;
f1 = f_ss(index1);f2 = f_ss(index2);

omega1 = 2*pi*f1; omega2 = 2*pi*f2;




% %Extracting Amplitude and Frequency
% max1 = max(y_meg);
% index1 = find(y_meg == max1);
% a1 = y_meg(index1);
% f1 = f(index1);
% y_meg(index1-3:index1+3) = NaN*ones(1,7);
% max2 = max(y_meg);
% index2 = find(y_meg == max2);
% a2 = y_meg(index2);
% f2 = f(index2);
% 
% 
% phase1 = angle(y_oneside(ceil(f1*maxtime)+1));
% phase2 = angle(y_oneside(ceil(f2*maxtime)+1));
% 
% omega1 = 2*pi*f1; omega2 = 2*pi*f2;
% 
% figure(4)
% plot(alltCM,xpeak_vals,'b')
% xlabel("$t$");ylabel("$x_{CM}$")
% axis([0 max(alltCM) -1.1*max(xpeak_vals) 1.1*max(xpeak_vals)])
% hold on
% %plot([0 max(alltCM)],[0,0],'k')
% fft_estimate = a1*cos(omega1*alltCM + phase1) + ...
%     a2*cos(omega2*alltCM+phase2);
% plot(alltCM,fft_estimate,'r')
% V_x = C0.*cos(omega.*alltCM+phi1+phi2);
% % plot(alltCM,V_x)
% legend("Soliton","Fourier Analysis","Potential")
% title(['$\omega_1 = ',num2str(omega1),...
%     ', \omega_2 = ',num2str(omega2),'$']);
% % 
% 
% 
% %Percent Error Estimate
% CurveDifference = norm(xpeak_vals-fft_estimate);
% PercentError = CurveDifference/norm(xpeak_vals);
% figure(5)
% error = abs(xpeak_vals-fft_estimate);
% avg_error = mean(error);
% plot(alltCM,error,'b')
% hold on
% plot([0,maxtime], [avg_error,avg_error],'r--')
% title(['Percent Error: ',num2str(PercentError)])
% xlabel("$t$");ylabel("$|\Delta|$");axis tight
% legend("Absolute Difference","Average")
% 
% 

%Curve Fitting
xdata = alltCM;
ydata = xpeak_vals;
fun = @(X,time)(X(1)*a1*cos(X(2)*omega1*time+X(3)*phase1)+...
    X(4)*a2*cos(X(5)*omega2*time + X(6)*phase2));
X0 = [1,1,1,1,1,1];

curvefit_parameters = lsqcurvefit(fun,X0,xdata,ydata);
%[X(1)*a1, X(2)*omega1,X(3)*phase1,X(4)*a2,X(5)*omega2,X(6)*phase2]


%Updating parameters
a1_cf = curvefit_parameters(1)*a1;
omega1_cf = curvefit_parameters(2)*omega1;
phase1_cf = curvefit_parameters(3)*phase1;

a2_cf = curvefit_parameters(4)*a2;
omega2_cf = curvefit_parameters(5)*omega2;
phase2_cf = curvefit_parameters(6)*phase2;


Curvefit_Estimate = a1_cf.*cos(omega1_cf.*alltCM+phase1_cf)+...
    a2_cf.*cos(omega2_cf.*alltCM+phase2_cf);



figure(6)
CurveDifference = norm(xpeak_vals-Curvefit_Estimate);
PercentError = CurveDifference/norm(xpeak_vals);
plot(alltCM,xpeak_vals,'b')
xlabel("$t$");ylabel("$x_{CM}$")
axis([0 max(alltCM) -1.1*max(xpeak_vals) 1.1*max(xpeak_vals)])
hold on
%plot([0 max(alltCM)],[0,0],'k')
plot(alltCM,Curvefit_Estimate,'r')
legend("Soliton","FFT + Curve Fit","Potential")
title(['$\omega_1 = ',num2str(omega1_cf),...
    ', \omega_2 = ',num2str(omega2_cf),', \omega_1 / \omega_p = '...
    ,num2str(omega1_cf/omega),'$']);
fprintf(['Percent Error:', num2str(PercentError),'\n'])

figure(7)
error_cf = abs(xpeak_vals-Curvefit_Estimate);
avg_error_cf = mean(error_cf);
plot(alltCM,error_cf,'b')
hold on
plot([0,maxtime], [avg_error_cf,avg_error_cf],'r--','LineWidth',2)
xlabel("$t$");ylabel("$|\Delta|$");axis tight
legend("Absolute Difference","Average")

fprintf(['Frequency Ratio = ', num2str(omega/omega1_cf),'\n']);
end


%Initial Soliton State
absu0 = abs(u00);
tol = 0.01;
%Find Width of the Soliton
%Find where the abs(u)>0.01, should be something like [0...0 1...1 0...0]
%Sum the resulting array to find the width in terms of indexes
%Split into 2 and create interval around the CM
SolitonIndexWidth = sum(absu0>tol);
IndexIntPM = ceil(SolitonIndexWidth/2);%Soliton Half Width in Indexes

usoliton = ones(2*IndexIntPM+1,nsaveCM);
xsoliton = usoliton;
%Initializing Soliton Center Index Array
IndexCM = zeros(1,nsaveCM+1);



%Initializing Overall mass array
Mass = zeros(1,nsaveCM+1);
Mass_soliton = Mass;

for ii = 1:nsaveCM+1
    %Overall mass 
    u_ii = alluCM(:,ii);
    mass = sum(u_ii.*conj(u_ii)).*dx;
    Mass(ii)=mass;

    %Soliton Window
    cmindex_ii = find(abs(x-xpeak_vals(ii))<tol);
    if isempty(cmindex_ii)
        cmindex_ii = IndexCM(ii-1);
    end
    IndexCM(ii) = cmindex_ii;
    usoliton(:,ii) = alluCM(IndexCM(ii)-IndexIntPM:IndexCM(ii)+...
        IndexIntPM,ii);
    xsoliton(:,ii) = x(IndexCM(ii)-IndexIntPM:IndexCM(ii)+...
        IndexIntPM);
    usoliton_ii = usoliton(ii);
    mass = sum(usoliton_ii.*conj(usoliton_ii)).*dx;
    Mass_soliton(ii) = mass;
end





%Plot conservation of mass:
figure(8)
mass0=Mass(1);
mass0_soliton = Mass_soliton(1);
plot(alltCM,(Mass-mass0)./mass0,'.-');
xlabel('$t$')
ylabel('$[M(t)-M(0)]/M(0)$')
figure(9)
plot(alltCM,(Mass_soliton-mass0_soliton)./mass0_soliton,'.-')
xlabel('$t$')
ylabel('$[M(t)-M(0)]/M(0)$')


figure(10)
plot(xsoliton(:,1),abs(usoliton(:,1)))
xmin = min(min(xsoliton));
xmax = max(max(xsoliton));
axis([xmin xmax -0.1*Maxmaxu 1.1*Maxmaxu])
xlabel("$x$");ylabel("$|u|$");
title(['$t=',num2str(alltCM(1)),'$']);
for ii = 2:nsaveCM+1
    plot(xsoliton(:,ii),abs(usoliton(:,ii)));
    axis([xmin xmax -0.1*Maxmaxu 1.1*Maxmaxu])
    xlabel("$x$");ylabel("$|u|$");title(['$t=',num2str(alltCM(ii)),'$']);
    drawnow;
end


