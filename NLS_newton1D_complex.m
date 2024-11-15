// Parameters
w = 0.5; // Temporal frequency of sought steady state
g = -1; // g = 1 is defocusing, g = -1 is focusing
L = -10; R = 10; // Left and right bounds of interval
Nx = 301; // Number of mesh points
x = linspace(L, R, Nx)'; // Discrete space
dx = x(2) - x(1); // Mesh size
ONE = ones(Nx, 1); // Unit vector for the discrete Laplacian
D2 = spdiags([ONE, -2 * ONE, ONE], -1:1, Nx, Nx); // Discrete Laplacian
D2(1, Nx) = 1; D2(Nx, 1) = 1; // Periodic Boundary Conditions
D2 = D2 / (dx^2);

// Indices for real and imaginary parts
indR = 1:Nx;
indI = Nx+1:2*Nx;

// Initial guess
A = sqrt(2 * w);
x0 = (R + L) / 2; // Amplitude and position of initial guess
u0 = A * sech(A * (x - x0)); // Unperturbed initial guess (sech soliton)

// Perturbation
rng('default'); // Reset random generator
pert = 0.3; // Size of perturbation
up = u0 + pert * (rand(Nx, 1) - 0.5) .* exp(-x.^2 / 10); // Perturbed initial condition
U = [real(up); imag(up)]; // Combine real and imaginary parts into one vector

// Newton method
it = 0;
err = 1; // Initializing error
if ~exist('V', 'var')
    V = 0; // If no potential, set V = 0
end

// Iteration loop
while err > 1e-10
    it = it + 1;
    Ur = U(indR);
    Ui = U(indI); // Real and imaginary parts of U

    // Modulus squared of u
    U2 = Ur.^2 + Ui.^2;

    // Jacobian matrix components
    J11 = -0.5 * D2 + diag(g * (3 * Ur.^2 + Ui.^2) + V + w);
    J22 = -0.5 * D2 + diag(g * (Ur.^2 + 3 * Ui.^2) + V + w);
    J12 = g * diag(2 * Ur .* Ui);

    // Full Jacobian matrix
    J = [J11, J12; J12, J22];

    // RHS of the system
    Fr = -0.5 * D2 * Ur + (g * U2 + V + w) .* Ur;
    Fi = -0.5 * D2 * Ui + (g * U2 + V + w) .* Ui;
    F = [Fr; Fi]; // Combine real and imaginary RHS

    // Newton correction
    DU = -J \ F;
    U1 = U + DU;
    err = norm(F); // Update error

    // Plotting progress of iteration
    figure(1);
    plot(x, U(indR), '.', x, U(indI), '.', x, U1(indR), x, U1(indI), x, V);
    xlabel('x');
    ylabel('u');
    title(['it=', num2str(it), ', error=', num2str(err)]);
    legend('Re(previous)', 'Im(previous)', 'Re(current)', 'Im(current)', 'V(x)', 'Location', 'NE');
    drawnow;
    fprintf('Press any key to continue...\n');
    pause;

    // Update U for the next iteration
    U = U1;
end

// Final solution
u = U(indR) + 1i * U(indI); // Combine real and imaginary parts into a complex vector
