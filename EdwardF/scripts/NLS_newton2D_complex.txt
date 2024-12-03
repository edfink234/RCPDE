function resid = nls2d_msd(psi, params)
    % Compute the residual for the 2D Nonlinear Schrödinger equation with
    % modulus-squared Dirichlet boundary conditions.

    % Unpack parameters
    npts_x = params.npts_x;
    npts_y = params.npts_y;
    dx = params.dx;
    dy = params.dy;
    V = reshape(params.V, npts_x, npts_y);
    mu = params.mu;

    % Pre-allocate nonlinear residual
    resid = zeros(2 * npts_x * npts_y, 1);

    % Decompose the field into real and imaginary parts
    X = reshape(psi(1:npts_x * npts_y), npts_x, npts_y);
    Y = reshape(psi(npts_x * npts_y + 1:end), npts_x, npts_y);

    % Compute the density and common term
    dens = X.^2 + Y.^2;
    comm = dens + V - mu;

    % Compute second derivatives
    d2Xdx2 = diff(X, 2, 1) / dx^2;
    d2Xdy2 = diff(X, 2, 2) / dy^2;
    d2Ydx2 = diff(Y, 2, 1) / dx^2;
    d2Ydy2 = diff(Y, 2, 2) / dy^2;

    % Total second derivatives
    d2X = d2Xdx2(:, 2:end-1) + d2Xdy2(2:end-1, :);
    d2Y = d2Ydx2(:, 2:end-1) + d2Ydy2(2:end-1, :);

    % Compute boundary terms
    term_l = (d2X(:, 1) .* X(2:end-1, 2) + d2Y(:, 1) .* Y(2:end-1, 2)) ./ dens(2:end-1, 2);
    term_r = (d2X(:, end) .* X(2:end-1, end-1) + d2Y(:, end) .* Y(2:end-1, end-1)) ./ dens(2:end-1, end-1);
    term_t = (d2X(1, :) .* X(2, 2:end-1) + d2Y(1, :) .* Y(2, 2:end-1)) ./ dens(2, 2:end-1);
    term_b = (d2X(end, :) .* X(end-1, 2:end-1) + d2Y(end, :) .* Y(end-1, 2:end-1)) ./ dens(end-1, 2:end-1);

    Omega_l = term_l - 2 * (dens(2:end-1, 2) - dens(2:end-1, 1) + V(2:end-1, 2) - V(2:end-1, 1));
    Omega_r = term_r - 2 * (dens(2:end-1, end-1) - dens(2:end-1, end) + V(2:end-1, end-1) - V(2:end-1, end));
    Omega_t = term_t - 2 * (dens(2, 2:end-1) - dens(1, 2:end-1) + V(2, 2:end-1) - V(1, 2:end-1));
    Omega_b = term_b - 2 * (dens(end-1, 2:end-1) - dens(end, 2:end-1) + V(end-1, 2:end-1) - V(end, 2:end-1));

    % Compute second derivatives at the endpoints
    Xdd_l = Omega_l .* X(2:end-1, 1);
    Xdd_r = Omega_r .* X(2:end-1, end);
    Xdd_t = Omega_t .* X(1, 2:end-1);
    Xdd_b = Omega_b .* X(end, 2:end-1);

    Ydd_l = Omega_l .* Y(2:end-1, 1);
    Ydd_r = Omega_r .* Y(2:end-1, end);
    Ydd_t = Omega_t .* Y(1, 2:end-1);
    Ydd_b = Omega_b .* Y(end, 2:end-1);

    % Pad second derivatives with zeros
    d2X = padarray(d2X, [1, 1], 0);
    d2Y = padarray(d2Y, [1, 1], 0);

    % Update second derivatives with boundary values
    d2X(1, 2:end-1) = Xdd_t;
    d2X(end, 2:end-1) = Xdd_b;
    d2X(2:end-1, 1) = Xdd_l;
    d2X(2:end-1, end) = Xdd_r;

    d2Y(1, 2:end-1) = Ydd_t;
    d2Y(end, 2:end-1) = Ydd_b;
    d2Y(2:end-1, 1) = Ydd_l;
    d2Y(2:end-1, end) = Ydd_r;

    % Compute corner terms using MSD boundary conditions
    assert(isequal(size(d2X), size(X)), 'd2X and X must have the same shape');
    assert(isequal(size(d2Y), size(Y)), 'd2Y and Y must have the same shape');
    assert(isequal(size(dens), size(X)) && isequal(size(dens), size(Y)), ...
           'dens must have the same shape as X and Y');
    
    term_tl = (d2X(1, 2) * X(1, 2) + d2Y(1, 2) * Y(1, 2)) / dens(1, 2);
    term_tr = (d2X(1, end-1) * X(1, end-1) + d2Y(1, end-1) * Y(1, end-1)) / dens(1, end-1);
    term_bl = (d2X(end, 2) * X(end, 2) + d2Y(end, 2) * Y(end, 2)) / dens(end, 2);
    term_br = (d2X(end, end-1) * X(end, end-1) + d2Y(end, end-1) * Y(end, end-1)) / dens(end, end-1);
    
    Omega_tl = term_tl - 2 * (dens(1, 2) - dens(1, 1) + V(1, 2) - V(1, 1));
    Omega_tr = term_tr - 2 * (dens(1, end-1) - dens(1, end) + V(1, end-1) - V(1, end));
    Omega_bl = term_bl - 2 * (dens(end, 2) - dens(end, 1) + V(end, 2) - V(end, 1));
    Omega_br = term_br - 2 * (dens(end, end-1) - dens(end, end) + V(end, end-1) - V(end, end));
    
    Xdd_tl_lr = Omega_tl * X(1, 1);
    Xdd_tr_lr = Omega_tr * X(end, 1);
    Xdd_bl_lr = Omega_bl * X(1, end);
    Xdd_br_lr = Omega_br * X(end, end);
    
    Ydd_tl_lr = Omega_tl * Y(1, 1);
    Ydd_tr_lr = Omega_tr * Y(end, 1);
    Ydd_bl_lr = Omega_bl * Y(1, end);
    Ydd_br_lr = Omega_br * Y(end, end);
    
    term_tl = (d2X(2, 1) * X(2, 1) + d2Y(2, 1) * Y(2, 1)) / dens(2, 1);
    term_tr = (d2X(2, end) * X(2, end) + d2Y(2, end) * Y(2, end)) / dens(2, end);
    term_bl = (d2X(end-1, 1) * X(end-1, 1) + d2Y(end-1, 1) * Y(end-1, 1)) / dens(end-1, 1);
    term_br = (d2X(end-1, end) * X(end-1, end) + d2Y(end-1, end) * Y(end-1, end)) / dens(end-1, end);
    
    Omega_tl = term_tl - 2 * (dens(2, 1) - dens(1, 1) + V(2, 1) - V(1, 1));
    Omega_tr = term_tr - 2 * (dens(2, end) - dens(1, end) + V(2, end) - V(1, end));
    Omega_bl = term_bl - 2 * (dens(end-1, 1) - dens(end, 1) + V(end-1, 1) - V(end, 1));
    Omega_br = term_br - 2 * (dens(end-1, end) - dens(end, end) + V(end-1, end) - V(end, end));
    
    Xdd_tl_tb = Omega_tl * X(1, 1);
    Xdd_tr_tb = Omega_tr * X(end, 1);
    Xdd_bl_tb = Omega_bl * X(1, end);
    Xdd_br_tb = Omega_br * X(end, end);
    
    Ydd_tl_tb = Omega_tl * Y(1, 1);
    Ydd_tr_tb = Omega_tr * Y(end, 1);
    Ydd_bl_tb = Omega_bl * Y(1, end);
    Ydd_br_tb = Omega_br * Y(end, end);
    
    d2X(1, 1) = (Xdd_tl_lr + Xdd_tl_tb) / 2.0;
    d2X(end, 1) = (Xdd_tr_lr + Xdd_tr_tb) / 2.0;
    d2X(1, end) = (Xdd_bl_lr + Xdd_bl_tb) / 2.0;
    d2X(end, end) = (Xdd_br_lr + Xdd_br_tb) / 2.0;
    
    d2Y(1, 1) = (Ydd_tl_lr + Ydd_tl_tb) / 2.0;
    d2Y(end, 1) = (Ydd_tr_lr + Ydd_tr_tb) / 2.0;
    d2Y(1, end) = (Ydd_bl_lr + Ydd_bl_tb) / 2.0;
    d2Y(end, end) = (Ydd_br_lr + Ydd_br_tb) / 2.0;

    % Return the system of nonlinear equations
    temp1 = -0.5 * d2X + comm .* X; % Intermediate result for real part
    temp2 = -0.5 * d2Y + comm .* Y; % Intermediate result for imaginary part
    resid(1:npts_x * npts_y) = temp1(:); % Flatten and assign to the first half of resid
    resid(npts_x * npts_y + 1:end) = temp2(:); % Flatten and assign to the second half of resid

end

% Parameters
params.npts_x = 101;
params.npts_y = 101;
params.dx = (10 - (-10)) / 100; % Mesh size in x direction
params.dy = (10 - (-10)) / 100; % Mesh size in y direction
params.mu = 2; % Temporal frequency = A^2
npts_x = params.npts_x;
npts_y = params.npts_y;

params.V = zeros(npts_x, npts_y); % Potential

x = linspace(-10, 10, npts_x);
y = linspace(-10, 10, npts_y);
[x_grid, y_grid] = meshgrid(x, y);

% Initial guess
w = params.mu;
A = sqrt(w);
x_0 = 1.0;  % Initial guess location
y_0 = 0.0;
dist = sqrt( ((x_grid-x_0).^2 + (y_grid-y_0).^2) );
r0 = 0.0; % Radius for initial guess
u0 = A * tanh(A * dist) .* exp(1j * atan2(y_grid-y_0, x_grid-x_0)); % Initial guess

% Perturbed initial guess
rng(0); % For reproducibility
pert = 2;
up = u0 + pert * (rand(npts_x, npts_y) - 0.5) .* exp(-(x_grid.^2 + y_grid.^2) / 10);
U = [real(up(:)); imag(up(:))];

% Newton's method
it = 0;
err = 1;
tol = 1e-4;

while err > tol
    it = it + 1;

    % Compute residual using a placeholder function for `nls2d_msd`
    F = nls2d_msd(U, params);

    % Apply modulus-squared Dirichlet boundary conditions
    Ur = reshape(U(1:npts_x * npts_y), npts_x, npts_y);
    Ui = reshape(U(npts_x * npts_y + 1:end), npts_x, npts_y);

    % Laplacian operators for 2D grid
    Ix = speye(npts_x);
    Iy = speye(npts_y);
    Dx = spdiags([ones(npts_x, 1), -2 * ones(npts_x, 1), ones(npts_x, 1)], [-1, 0, 1], npts_x, npts_x) / params.dx^2;
    Dy = spdiags([ones(npts_y, 1), -2 * ones(npts_y, 1), ones(npts_y, 1)], [-1, 0, 1], npts_y, npts_y) / params.dy^2;
    Laplacian = kron(Iy, Dx) + kron(Dy, Ix);

    % Diagonal terms for J11 and J22
    dens = Ur.^2 + Ui.^2;
    V_flat = params.V(:);
    J11 = -0.5 * Laplacian + spdiags(3 * Ur(:).^2 + Ui(:).^2 + V_flat - params.mu, 0, npts_x * npts_y, npts_x * npts_y);
    J22 = -0.5 * Laplacian + spdiags(3 * Ui(:).^2 + Ur(:).^2 + V_flat - params.mu, 0, npts_x * npts_y, npts_x * npts_y);

    % Off-diagonal terms J12 and J21
    J12 = spdiags(2 * Ur(:) .* Ui(:), 0, npts_x * npts_y, npts_x * npts_y);
    J21 = spdiags(2 * Ur(:) .* Ui(:), 0, npts_x * npts_y, npts_x * npts_y);

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
u = reshape(U(1:npts_x * npts_y) + 1j * U(npts_x * npts_y + 1:end), npts_x, npts_y);

% Plot the final solution
figure;
surf(x_grid, y_grid, abs(u), 'EdgeColor', 'none');
colormap default
a = colorbar;
a.Label.String = '|u|';
xlabel('x');
ylabel('y');
zlabel('|u|');
title('Final Solution Magnitude');
view(3);
