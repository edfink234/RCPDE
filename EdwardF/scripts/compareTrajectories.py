import numpy as np
import matplotlib.pyplot as plt
from LearnInvertedPenulum import master_func_learn_ivp_ode
from LearnInvertedPenulumPDE import master_func_learn_ivp_pde
from os import system

def compare_ode_pde_func(v_start = 2):
    T=100
    xStart = 2.5
    v_start = v_start
    A_ode, b_ode = 0.675, 0.666
    A_pde, b_pde = 1, 1

    x_pde, t_pde, xStart, mass_values = master_func_learn_ivp_pde(simulate_only = {"simulate_only": True, "xStart": xStart, "store mass values": True}, T = T, A = A_pde, b = b_pde, v_start = v_start, interpolate = False, add_kick = True)
    x_ode, t_ode = master_func_learn_ivp_ode(simulate_only = {"simulate_only": True, "xStart": xStart}, T = T, A = A_ode, b = b_ode, v_start = v_start)

    assert(t_ode[0] == t_pde[0] and t_ode[-1] == t_pde[-1])

    #TODO: Start further from the potential minima
    #TODO: consider adding a kick (u_converged_from_newton *= exp(i*c*x)), trial and error with velocity c

    plt.plot(t_ode, x_ode, label = "ODE")
    plt.plot(t_pde, x_pde, label = "PDE")
    plt.xlabel("t")
    plt.ylabel("x")

    plt.legend()
    plt.title(f"Trajectory ODE & PDE $x_0$ = {xStart:.4f}, $v_0$ = {v_start:0.4f}")
    plt.savefig(f"trajectory_data_ode_and_pde_T_{T}_A_{A_ode},_b_{b_ode}_xStart_{xStart}_vStart_{v_start}.png", dpi=5*96)
    plt.close()
    system(f"open trajectory_data_ode_and_pde_T_{T}_A_{A_ode},_b_{b_ode}_xStart_{xStart}_vStart_{v_start}.png")

    plt.plot(t_pde, mass_values, label = "PDE $|u|^2 Loss$")
    assert(len(mass_values) == len(t_pde))
#    print(f"mass_values = {mass_values}")
    plt.xlabel("t")
    plt.ylabel(r"$|u|^2$ Loss")
    plt.legend()
    plt.title(f"PDE Mass Loss $x_0$ = {xStart:.4f}, $v_0$ = {v_start:0.4f}")
    plt.savefig(f"mass_loss_pde_T_{T}_A_{A_ode},_b_{b_ode}_xStart_{xStart}_vStart_{v_start}.png", dpi=5*96)
    plt.close()
    system(f"open mass_loss_pde_T_{T}_A_{A_ode},_b_{b_ode}_xStart_{xStart}_vStart_{v_start}.png")

for i in range(1, 11):
    compare_ode_pde_func(v_start = i)
