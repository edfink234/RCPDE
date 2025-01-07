function [V] = TDPotential(x,t,params)
%params is the parameter vector and is structured as below
%params = [A0,B0,C0,nu,gamma,omega,phi1,phi2]';
%A0 \in [0,1]
%B0 \in [0,0.5]
%C0 \in [0,10]
%[nu,gamma,omega] \in [0,50]
%[phi1,phi2] \in [0,pi]
A0 = params(1);B0 = params(2); C0 = params(3);
nu = params(4); gamma = params(5); omega = params(6);
phi1 = params(7); phi2 = params(8);
A = A0*(cos(nu*t));%Height and freq. of potential 
B = 1+B0*cos(gamma*t+phi1);%Width of potential
xi = C0*cos(omega*t+phi1+phi2);%Width and frequency of pot. Osc.s
V = A.*(sech(B.*(x-xi))).^2;
end