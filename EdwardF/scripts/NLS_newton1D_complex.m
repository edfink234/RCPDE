% Parameters
params.nls.npts = 301;  % Number of grid points
params.nls.dx = (10 - (-10)) / 300;  % Mesh size
params.nls.V = zeros(params.nls.npts, 1);  % Potential
params.nls.mu = 1;  % Temporal frequency
params.nls.g = 1;  % Defocusing parameter
npts = params.nls.npts;

% Spatial grid
L = -10; R = 10;
x = linspace(L, R, npts)';

% Initial guess
w = params.nls.mu;
A = sqrt(2 * w);
x0 = (R + L) / 2;  % Center of initial guess
u0 = A * tanh(A * (x - x0));  % Initial guess (tanh soliton)

% Perturbed initial guess
rng(0);  % For reproducibility
pert = 0;
up = u0 + pert * (rand(npts, 1) - 0.5) .* exp(-x.^2 / 10);
U = [real(up); imag(up)];  % Real and imaginary parts concatenated

% Newton's method
it = 0;
err = 1;
tol = 1e-9;

while err > tol
    it = it + 1;
    
    % Compute residual using nls1d_msd
    F = nls1d_msd(U, params);
    
    % Decompose U into real and imaginary parts
    Ur = U(1:npts);
    Ui = U(npts+1:end);
    dx = params.nls.dx;
    V = params.nls.V;
    mu = params.nls.mu;

    % Finite difference Laplacian for a 1D grid
    D2 = spdiags([ones(npts, 1), -2*ones(npts, 1), ones(npts, 1)], -1:1, npts, npts) / dx^2;

    % Diagonal terms for J11 and J22
    J11 = -0.5 * D2 + spdiags(2*Ur.^2 + V - mu, 0, npts, npts);  % Real part X
    J22 = -0.5 * D2 + spdiags(2*Ui.^2 + V - mu, 0, npts, npts);  % Imaginary part Y

    % Off-diagonal terms J12 and J21
    J12 = spdiags(2 * Ur .* Ui, 0, npts, npts);
    J21 = spdiags(2 * Ur .* Ui, 0, npts, npts);

    % Assemble the full Jacobian as a block matrix
    J = [J11, J12; J21, J22];

    % Newton correction
    DU = J \ (-F);
    U1 = U + DU;
    err = norm(F);
    fprintf('Iteration %d, Error: %.2e\n', it, err);
    
    % Update U
    U = U1;
end

% Final solution
u = U(1:npts) + 1i * U(npts+1:end);

% Plot the final solution
figure;
plot(x, real(u), 'DisplayName', 'Re(u)');
hold on;
plot(x, imag(u), 'DisplayName', 'Im(u)');
xlabel('x');
ylabel('u');
title('Final Solution');
legend;
grid on;

function resid = nls1d_msd(psi, params)
    % Unpack parameters
    npts = params.nls.npts;
    dx = params.nls.dx;
    V = params.nls.V;
    mu = params.nls.mu;
    
    % Pre-allocate nonlinear residual
    resid = zeros(2 * npts, 1);
    
    % Decompose the field into real and imaginary parts
    X = psi(1:npts);
    Y = psi(npts+1:end);
    
    % Compute the density and common term
    dens = X.^2 + Y.^2;
    comm = dens + V - mu;
    
    % Compute the 1D Laplacians inside the domain (finite differences)
    d2Xdx2 = diff(X,2) / dx^2;
    d2Ydx2 = diff(Y,2) / dx^2;

    % Compute the common term (see, the term in the square brackets in Eq.
    % (3.4) in Hermano Ricardo's paper)--I am calling it \Omega:
    term_l = ( d2Xdx2(1).*X(2)+d2Ydx2(1).*Y(2) ) / dens(2);
    term_r = ( d2Xdx2(npts-2).*X(npts-1)+d2Ydx2(npts-2).*Y(npts-1) ) / dens(npts-1);
    Omega_l = term_l - 2 * ( dens(2) - dens(1) + V(2) - V(1) );
    Omega_r = term_r - 2 * ( dens(npts-1) - dens(npts) + V(npts-1) - V(npts) );

    % Compute the 2nd-order derivatives of X and Y and the endpoints:
    Xdd_l = Omega_l * X(1); Xdd_r = Omega_r * X(npts);
    Ydd_l = Omega_l * Y(1); Ydd_r = Omega_r * Y(npts);

    % Update the pre-computed derivatives/Concatenate the vectors:
    d2Xdx2 = [ Xdd_l; d2Xdx2; Xdd_r ];
    d2Ydx2 = [ Ydd_l; d2Ydx2; Ydd_r ];
    
    % Return the system of nonlinear equations
    resid(1:npts) = -0.5 * d2Xdx2 + comm .* X;
    resid(npts+1:end) = -0.5 * d2Ydx2 + comm .* Y;
end

