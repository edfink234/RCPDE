import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============================================================
# Paste your best SR postfix expression here.
#
# Example:
#     SR_XI_POSTFIX = "x0 0.2 * tanh 2 *"
#
# Meaning:
#     xi(t) = 2*tanh(0.2*t)
# ============================================================

SR_XI_POSTFIX = "-10.00394920949605 0.0009615554218509459 x0 + + x0 sin 4.881411772212712 x0 * + * 0.035117916693453655 *"


# ============================================================
# PDE/control parameters
# ============================================================

Omega = 0.18
A = 1.0
b = 0.75
A_sol = 0.75
g = -1.0

T = 10.0
dt = 0.001
num_steps = int(T / dt)
t_values = np.linspace(1e-8, T, num_steps)

L = -10.0
R = 10.0
N_raw = 401

x = np.linspace(L, R, N_raw)[:-1]
N = len(x)
dx = x[1] - x[0]

lap_coeff = 0.5 / (dx * dx)
one_over_six = 1.0 / 6.0

x_start = 2.7615
v_start = 0.0
x_star = 0.0

# Use every `stride`-th PDE frame in the movie.
# stride = 1 gives a huge movie.
# stride = 10 or 20 is usually enough.
stride = 10

out_movie = Path("sr_bright_soliton_control.mp4")
out_plot = Path("sr_bright_soliton_trajectory.png")


# ============================================================
# Small postfix evaluator for SR expressions
# ============================================================

def sech(z):
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / np.cosh(z)


def safe_div(a, b):
    if abs(b) < 1e-12:
        return 1e12 if a >= 0 else -1e12
    return a / b


def safe_pow(a, b):
    try:
        y = a ** b
        if not np.isfinite(y):
            return 1e12
        return y
    except Exception:
        return 1e12


def eval_postfix(expr, t):
    stack = []

    for tok in expr.split():
        if tok == "x0":
            stack.append(float(t))

        elif tok in {"+", "-", "*", "/", "^"}:
            if len(stack) < 2:
                raise ValueError(f"Bad postfix expression near binary operator {tok!r}")

            rhs = stack.pop()
            lhs = stack.pop()

            if tok == "+":
                stack.append(lhs + rhs)
            elif tok == "-":
                stack.append(lhs - rhs)
            elif tok == "*":
                stack.append(lhs * rhs)
            elif tok == "/":
                stack.append(safe_div(lhs, rhs))
            elif tok == "^":
                stack.append(safe_pow(lhs, rhs))

        elif tok in {"~", "neg"}:
            if not stack:
                raise ValueError(f"Bad postfix expression near unary operator {tok!r}")
            stack.append(-stack.pop())

        elif tok == "sin":
            stack.append(math.sin(stack.pop()))
        elif tok == "cos":
            stack.append(math.cos(stack.pop()))
        elif tok == "tan":
            stack.append(math.tan(stack.pop()))
        elif tok == "tanh":
            stack.append(math.tanh(stack.pop()))
        elif tok == "sech":
            stack.append(float(sech(stack.pop())))
        elif tok in {"asin", "arcsin"}:
            stack.append(math.asin(np.clip(stack.pop(), -1.0, 1.0)))
        elif tok in {"acos", "arccos"}:
            stack.append(math.acos(np.clip(stack.pop(), -1.0, 1.0)))
        elif tok in {"ln", "log"}:
            a = stack.pop()
            stack.append(math.log(abs(a) + 1e-12))
        elif tok == "exp":
            stack.append(math.exp(np.clip(stack.pop(), -50.0, 50.0)))
        elif tok == "sqrt":
            stack.append(math.sqrt(abs(stack.pop())))

        else:
            try:
                stack.append(float(tok))
            except ValueError as exc:
                raise ValueError(f"Unknown token in SR expression: {tok!r}") from exc

    if len(stack) != 1:
        raise ValueError(f"Postfix expression ended with stack size {len(stack)}, not 1.")

    y = stack[0]

    if not np.isfinite(y):
        return 1e12

    return float(y)


def xi_sr(t):
    return eval_postfix(SR_XI_POSTFIX, t)


# ============================================================
# PDE helpers
# ============================================================

def potential(x_grid, xi):
    return 0.5 * Omega * Omega * x_grid * x_grid + A * sech(b * (x_grid - xi)) ** 2


def nls_rhs(u, V):
    up = np.roll(u, 1)
    um = np.roll(u, -1)

    return 1j * (
        lap_coeff * (up - 2.0 * u + um)
        - (g * np.abs(u) ** 2 + V) * u
    )


def rk4_step(u, V):
    k1 = dt * nls_rhs(u, V)
    k2 = dt * nls_rhs(u + 0.5 * k1, V)
    k3 = dt * nls_rhs(u + 0.5 * k2, V)
    k4 = dt * nls_rhs(u + k3, V)

    return u + one_over_six * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def center_of_mass(u):
    rho = np.abs(u) ** 2

    trapz = getattr(np, "trapezoid", np.trapz)

    den = trapz(rho, x)

    if den <= 0.0 or not np.isfinite(den):
        return np.nan

    num = trapz(x * rho, x)
    return num / den


def initial_state():
    state_path = Path("bright_soliton_newton_state.csv")

    if state_path.exists():
        data = np.genfromtxt(
            state_path,
            delimiter=",",
            comments="#",
            names=True,
            skip_header = 18
        )

        x_loaded = np.asarray(data["x"], dtype=np.float64)
        u_loaded = (
            np.asarray(data["u_real"], dtype=np.float64)
            + 1j * np.asarray(data["u_imag"], dtype=np.float64)
        )

        if len(x_loaded) != len(x):
            raise ValueError(
                f"Loaded Newton state has length {len(x_loaded)}, "
                f"but movie grid has length {len(x)}."
            )

        if not np.allclose(x_loaded, x, atol=1e-9, rtol=1e-12):
            max_err = np.max(np.abs(x_loaded - x))
            raise ValueError(
                f"Loaded Newton state grid does not match movie grid. "
                f"max grid error = {max_err}"
            )

        print(f"Loaded Newton state from {state_path}")
        return u_loaded.astype(np.complex128)

    print("No Newton CSV found; using analytic sech initial condition.")
    return A_sol * sech(A_sol * (x - x_start)).astype(np.complex128)


# ============================================================
# Simulate and store frames
# ============================================================

u = initial_state()

saved_t = []
saved_xi = []
saved_xcm = []
saved_density = []


def save_frame(t):
    xi = xi_sr(t)
    xcm = center_of_mass(u)
    rho = np.abs(u) ** 2

    saved_t.append(t)
    saved_xi.append(xi)
    saved_xcm.append(xcm)
    saved_density.append(rho.copy())


save_frame(t_values[0])

for i in range(1, num_steps):
    t = t_values[i]
    xi = xi_sr(t)
    V = potential(x, xi)

    u = rk4_step(u, V)

    if not np.all(np.isfinite(u)):
        raise RuntimeError(f"PDE blew up at i={i}, t={t}")

    if i % stride == 0 or i == num_steps - 1:
        save_frame(t)

saved_t = np.asarray(saved_t)
saved_xi = np.asarray(saved_xi)
saved_xcm = np.asarray(saved_xcm)
saved_density = np.asarray(saved_density)


# ============================================================
# Compute approximate velocity and best synchronization time
# ============================================================

saved_v = np.gradient(saved_xcm, saved_t)
saved_a = np.gradient(saved_v, saved_t)

sync_score = saved_xcm ** 2 + saved_v ** 2 + saved_xi ** 2

# Avoid the first couple of frames when locating the best time.
start_idx = min(2, len(sync_score) - 1)
best_idx = start_idx + int(np.argmin(sync_score[start_idx:]))

print("Best approximate synchronization:")
print(f"  t       = {saved_t[best_idx]:.6f}")
print(f"  x_cm    = {saved_xcm[best_idx]:.6e}")
print(f"  v       = {saved_v[best_idx]:.6e}")
print(f"  xi      = {saved_xi[best_idx]:.6e}")
print(f"  score   = {sync_score[best_idx]:.6e}")


# ============================================================
# Save trajectory plot
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    saved_t,
    saved_xcm,
    label=r"$x_{\mathrm{cm}}(t)$ [m]",
    color="blue"
)

plt.plot(
    saved_t,
    saved_v,
    label=r"$v(t)$ [m/s]",
    color="green",
    linestyle=":"
)

plt.plot(
    saved_t,
    saved_a,
    label=r"$a(t)$ [m/s$^2$]",
    color="purple",
    linestyle="-."
)

plt.plot(
    saved_t,
    saved_xi,
    label=r"$\xi(t)$",
    color="red",
    linestyle="--"
)

plt.axhline(
    y=0.0,
    color="black",
    linestyle="--",
    alpha=0.2,
    linewidth=0.5
)

plt.axvline(
    saved_t[best_idx],
    linestyle=":",
    linewidth=0.8,
    alpha=0.5
)

print(f"Saved trajectory plot to {out_plot}")

plt.xlabel("t")
plt.ylabel("position")
plt.title("SR-discovered control trajectory\n"rf"$A_{{sol}}={A_sol:.2f},\,A={A:.2f},\,b={b:.2f},\,\Omega={Omega:.2f}$")
plt.legend()
plt.tight_layout()
plt.savefig(out_plot, dpi=200)
plt.close()

print(f"Saved trajectory plot to {out_plot}")

# ============================================================
# Make movie
# ============================================================

fig, ax = plt.subplots(figsize=(8, 6))

density_line, = ax.plot([], [], lw=1.5, label=r"$|u(x,t)|^2$")
potential_line, = ax.plot([], [], lw=2.0, label=r"$V(x,t)$")
center_dot, = ax.plot([], [], "o", markersize=7, label="center of mass")
target_dot, = ax.plot([0.0], [0.0], "x", markersize=7, label=r"$x^\star=0$")

ax.set_xlim(x[0], x[-1])

sampled_vmax = 0.0
for k in range(0, len(saved_xi), max(1, len(saved_xi) // 50)):
    sampled_vmax = max(sampled_vmax, float(np.max(potential(x, saved_xi[k]))))

rho_max = float(np.max(saved_density))
ymax = 1.15 * max(rho_max, sampled_vmax, 1.0)

ax.set_ylim(0.0, ymax)
ax.set_xlabel("x")
ax.set_ylabel("density / potential")
ax.legend(loc="upper right")


def init():
    density_line.set_data([], [])
    potential_line.set_data([], [])
    center_dot.set_data([], [])
    return density_line, potential_line, center_dot, target_dot


def update(k):
    t = saved_t[k]
    xi = saved_xi[k]
    xcm = saved_xcm[k]
    rho = saved_density[k]
    V = potential(x, xi)

    density_line.set_data(x, rho)
    potential_line.set_data(x, V)

    center_y = potential(np.asarray([xcm]), xi)[0]
    center_dot.set_data([xcm], [center_y])

    ax.set_title(
        rf"SR control: $t={t:.3f}$, "
        rf"$x_{{cm}}={xcm:.3f}$, "
        rf"$\xi={xi:.3f}$"
    )

    return density_line, potential_line, center_dot, target_dot


fps = 30

ani = animation.FuncAnimation(
    fig,
    update,
    frames=len(saved_t),
    init_func=init,
    blit=True,
    interval=1000.0 / fps,
)

try:
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(out_movie, writer=writer)
    print(f"Saved movie to {out_movie}")
except Exception as exc:
    gif_path = out_movie.with_suffix(".gif")
    print(f"Could not save mp4 because: {exc}")
    print(f"Trying gif instead: {gif_path}")
    writer = animation.PillowWriter(fps=fps)
    ani.save(gif_path, writer=writer)
    print(f"Saved movie to {gif_path}")

plt.close()
