%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% NLS_newton1D_complex.m:
% Using Newton's method to compute a COMPLEX steady state for NLS:
%    i u_t = -(1/2) u_xx + g*|u|^2 u + V(x)*u
% Ricardo Carretero, Panos Kevrekidis, Dimitri Frantzeskakis, 2024.
% Code available at:
%    http://nonlinear.sdsu.edu/~carreter/NonlinearWavesBook/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(exist('itmax')==0)
    itmax=10;
end
if ~exist('use_built_in_fitter','var')
    use_built_in_fitter = false;    % toggle this to true to use fsolve
end

itmax = 9000;
ONE=ones(N,1);         % unit vector for the discrete Laplacian
D2=spdiags([ONE,-2*ONE,ONE],-1:1,N,N); % Discrete Laplacian
D2(1,N)=1; D2(N,1)=1;                  % Periodic Boundary Conds
D2=D2/(dx^2);
indR=1:N; indI=N+1:2*N;               % index for real & imag
err_tol = 1e-10;

if(exist('V')==0) V=0; end;  % If no potential => V=0

if use_built_in_fitter

    % wrap all parameters into the residual function
    fun = @(U) myResidual(U, D2, g, V, wfreq, N);

    % set fsolve options to roughly match your Newton tolerances
    opts = optimoptions('fsolve', ...
                        'Display','iter', ...
                        'TolFun',err_tol, ...
                        'MaxIterations',itmax);

    % call the solver (initial guess is your current U)
    [U, Fval, exitflag, output] = fsolve(fun, U, opts);

    % compute and print the final error
    err = norm(Fval);
    assert(err < err_tol, strcat("ASSERT ERROR: solution did not converge for wfreq = ", num2str(wfreq)));
    fprintf('Built-in solver exitflag = %d, iterations = %d, final err = %e\n', ...
            exitflag, output.iterations, err);

     if plot_newton
         figure(newton_figure);                  % Plotting progress of iteration
         plot(x,U(indR),'.',x,U(indI),'.',x,V)
         xlabel('x'); ylabel('u');
         title(['# of iterations = ',num2str(output.iterations),', error=',num2str(err)]);
         legend('Re($u_{\mathrm{converged}}$)','Im($u_{\mathrm{converged}}$)','V(x)','Location','NE', 'Interpreter', 'latex');
         drawnow; %fprintf('Press any key to continue...\n'); pause
     end

else

    it=0; err=1;             % initializing error

    while((err>err_tol)&(it<itmax)) % Main loop: checking Newton tolerance
     it=it+1;
     
     Ur=U(indR); Ui=U(indI);     % real and imag parts of u
    
     U2 = Ur.^2+Ui.^2;                              % mod square of u
     J11=-0.5*D2+diag(g*(3*Ur.^2+  Ui.^2)+V+wfreq); % J11 part of Jacobian
     J22=-0.5*D2+diag(g*(  Ur.^2+3*Ui.^2)+V+wfreq); % J22 part of Jacobian
     J12=g*diag(2*Ur.*Ui);                          % J12 part of Jacobian
     J = [J11,J12; J12,J22];                        % Full Jacobian
     Fr = -0.5*D2*Ur+(g*U2+V+wfreq).*Ur;            % real(RHS)
     Fi = -0.5*D2*Ui+(g*U2+V+wfreq).*Ui;            % imag(RHS)
     F = [Fr; Fi];                                  % RHS
     
     DU = -J\F;                  % Newton correction
     U1 = U+DU;                  % New step through Newton
     err=norm(F);                % How close to convergence we are
    
     if plot_newton
         figure(newton_figure);                  % Plotting progress of iteration
         plot(x,U(indR),'.',x,U(indI),'.',x,U1(indR),x,U1(indI),x,V)
         xlabel('x'); ylabel('u');
         title(['it=',num2str(it),', error=',num2str(err)]);
         legend('Re(previous)','Im(previous)','Re(current)',...
                'Im(current)','V(x)','Location','NE')
         drawnow; %fprintf('Press any key to continue...\n'); pause
     end
    
     U = U1;                     % Update solution
    end
    fprintf("Error after Newton for wfreq = %e = %e\n", wfreq, err);
end
u = U(indR)+1i*U(indI);      % wrapping into a complex vector
assert(err < err_tol, strcat("ASSERT ERROR: solution did not converge for wfreq = ", num2str(wfreq)));


function F = myResidual(U, D2, g, V, wfreq, N)
    % unpack
    Ur = U(1:N);
    Ui = U(N+1:2*N);
    U2 = Ur.^2 + Ui.^2;
    % same residuals as before
    Fr = -0.5*D2*Ur + (g*U2 + V + wfreq).*Ur;
    Fi = -0.5*D2*Ui + (g*U2 + V + wfreq).*Ui;
    F  = [Fr; Fi];
end
