import numpy as np
import matplotlib.pyplot as plt
from LearnInvertedPenulum import master_func_learn_ivp_ode
from LearnInvertedPenulumPDE import master_func_learn_ivp_pde
from os import system

T=100
xStart = 2.5
A_ode, b_ode = 0.675, 0.666
A_pde, b_pde = 1, 1

x_ode, t_ode = master_func_learn_ivp_ode(simulate_only = {"simulate_only": True, "xStart": xStart}, T = T, A = A_ode, b = b_ode)
x_pde, t_pde = master_func_learn_ivp_pde(simulate_only = {"simulate_only": True, "xStart": xStart}, T = T)

assert(t_ode[0] == t_pde[0] and t_ode[-1] == t_pde[-1])

plt.plot(t_ode, x_ode, label = "ODE")
plt.plot(t_pde, x_pde, label = "PDE")
plt.xlabel("t")
plt.ylabel("x")

plt.legend()
plt.title(f"Trajectory ODE & PDE $x_0$ = {xStart}")
plt.savefig(f"trajectory_data_ode_and_pde_T_{T}_A_{A_ode},_b_{b_ode}.png", dpi=5*96)
plt.close()
system(f"open trajectory_data_ode_and_pde_T_{T}_A_{A_ode},_b_{b_ode}.png")

