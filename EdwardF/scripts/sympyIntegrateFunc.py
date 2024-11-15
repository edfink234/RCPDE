#from sympy import symbols, integrate, sech, oo
#
## Define the symbols for the variables
#x, A, B, D, E, t = symbols('x A B D E t', real=True)
#
## Define the integrand using the hyperbolic secant squared functions
#integrand = sech(A - E*x)**2 * sech(B - D*x)**2
#
## Perform the integration with respect to x over the range from -∞ to ∞
#result = integrate(integrand, (x, -oo, oo))
#
## Display the result
#print(result)
from scipy.integrate import quad
import numpy as np

sech = lambda x: 1/np.cosh(x)

# Define a lambda function for the integrand (substitute values for parameters if needed)
integrand_func = lambda x: sech(A - E * x)**2 * sech(B - D * x)**2

# Perform numerical integration over the range -∞ to ∞
result, error = quad(integrand_func, -np.inf, np.inf)
print(result)
