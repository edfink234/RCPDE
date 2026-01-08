# RCPDE — Bright Soliton Control & Reduced-Order PDE Comparisons

This repository contains code and data supporting an investigation of
the control of bright solitons governed by the nonlinear Schrödinger equation (NLS) using
a variational reduced model and neural-network based controller.

The goal of the investigation is to test whether a controller learned on a reduced point-particle
approximation transfers to the full PDE dynamics, and to document where that approximation
breaks down. The code here implements:
- the PDE simulation with periodic boundary conditions,
- the RK4 time integrator and loss functions,
- tools for generating and visualizing soliton trajectories,
- and scripts to produce the figures referenced in the chapter. 
