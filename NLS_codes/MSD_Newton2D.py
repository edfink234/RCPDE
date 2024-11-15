import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Lx, Ly = 10.0, 10.0        # Domain size in x and y directions
Nx, Ny = 100, 100          # Number of grid points
T = 5.0                    # Total simulation time
dt = 0.0001                 # Time step size
steps = int(T / dt)        # Number of time steps
dx = Lx / Nx               # Grid spacing in x
dy = Ly / Ny               # Grid spacing in y

# Create grid
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
X, Y = np.meshgrid(x, y)

# Initialize fields: u_real and u_imag are real and imaginary parts of u
u_real = np.exp(-(X**2 + Y**2))  # Example initial condition (Gaussian)
u_imag = np.zeros((Nx, Ny))      # Imaginary part starts at zero

# Laplacian operator function
def laplacian(Z, dx, dy):
    Zxx = (np.roll(Z, -1, axis=0) - 2 * Z + np.roll(Z, 1, axis=0)) / dx**2
    Zyy = (np.roll(Z, -1, axis=1) - 2 * Z + np.roll(Z, 1, axis=1)) / dy**2
    return Zxx + Zyy

# Time-stepping loop
for n in range(steps):
    # Compute the squared magnitude of u
    magnitude_squared = u_real**2 + u_imag**2

    # Update u_real and u_imag using the discretized NLS equations
    u_real_new = u_real + dt * (0.5 * laplacian(u_imag, dx, dy) + magnitude_squared * u_imag)
    u_imag_new = u_imag - dt * (0.5 * laplacian(u_real, dx, dy) + magnitude_squared * u_real)

    # Update fields
    u_real, u_imag = u_real_new, u_imag_new

    # Optional: plot the magnitude of u every few steps
    if n % 100 == 0:
        plt.clf()
        plt.imshow(np.sqrt(u_real**2 + u_imag**2), extent=(-Lx/2, Lx/2, -Ly/2, Ly/2))
        plt.colorbar()
        plt.title(f'Time: {n * dt:.2f}')
        plt.pause(0.01)

plt.show()


