import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
A_sol = 1.0
x0 = 0.0
c = 1.0

# Spatial domain
x = np.linspace(-10, 10, 1000)

# Times to plot
times = [0.0, 1.0, 2.0]

# Define sech function
def sech(z):
    return 1 / np.cosh(z)

# Create figure with 3 rows and 1 column
fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

for ax, t in zip(axes, times):
    # Bright soliton |u|^2
    sech_term = sech(A_sol * (x - c*t - x0))
    comm_term = A_sol*sech_term
    
    u_sq = comm_term*comm_term
    trig_arg = c*x + (A_sol*A_sol - c*c)*0.5*t
    u_re = comm_term*np.cos(trig_arg)
    u_im = comm_term*np.sin(trig_arg)
    
    # Plotting
    ax.plot(x, u_sq, label = "$|u|^2$")
    ax.plot(x, u_re, label = "$\mathrm{Re}(u)$")
    ax.plot(x, u_im, label = "$\mathrm{Im}(u)$")
    ax.set_title(f'$t = {t}$')
    ax.legend()
    ax.grid(True)

# Common x-label
axes[-1].set_xlabel('x')

# compute each axes' center in figure coords
centers = []
for ax in axes:
    pos = ax.get_position()            # Bbox in figure coords
    centers.append(pos.x0 + pos.width/2)

# take the mean (or pick (first+last)/2)
x_center = sum(centers) / len(centers)

fig.suptitle(r"Free Bright Soliton: $|u|^2$, Re($u$), Im($u$)"+"\n\n"+r"($A_{\mathrm{sol}} =$"f"{A_sol}, "r"$x_0 =$"f"{x0}, "+r"$c=$"+f"{c})", x=x_center+.02)

plt.tight_layout()
plt.savefig("bright_soliton_V_equals_0_mod_squared_re_u_im_u.svg")
os.system(f"rsvg-convert -f pdf -o bright_soliton_V_equals_0_mod_squared_re_u_im_u.pdf bright_soliton_V_equals_0_mod_squared_re_u_im_u.svg")
os.system("rm bright_soliton_V_equals_0_mod_squared_re_u_im_u.svg")
os.system("open bright_soliton_V_equals_0_mod_squared_re_u_im_u.pdf")

