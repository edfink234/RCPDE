from sympy import symbols, Function, Eq, dsolve, Symbol, latex, sin, cos, Derivative, series, simplify, pi, diff

def ODE_IVP():
    r'''
    Solves the ODE:
        m\ddot{x} = -\Omega^2 x + \Omega^2 \xi(t)
    '''
    # Define symbols and functions
    t, Omega = symbols('t Omega')
    m = Symbol('m', positive = True)
    x = Function('x')(t)
    xi = Function('xi')(t)

    # Define the differential equation
    ode = Eq(m * x.diff(t, t), -Omega**2 * x + Omega**2 * xi)

    # Solve the differential equation
    solution = dsolve(ode, x)
    print(latex(solution).replace(r"\int", r"\displaystyle \int"))

def ODE_IVP_PLUS_SOLITON():
    r'''
    Solves the ODE system:
    
        m\ddot{x} = -{\Omega}^{2} x + \Omega^2\xi(t)
        \Ddot{\xi} + \frac{\Omega^2}{2}\xi = 0
        
    '''
    # Define symbols and functions
    Omega = Symbol('Omega', positive=True)
    t = Symbol('t', positive=True)
    m = Symbol('m', positive=True)
    x = Function('x')(t)
    xi = Function('xi')(t)

    # Define the ODE for xi(t) (Bob's ODE)
    ode_xi = Eq(xi.diff(t, t) + (Omega**2 / 2) * xi, 0)

    # Solve the ODE for xi(t) (optional, for verification purposes)
    solution_xi = dsolve(ode_xi, xi)

    # Define the ODE for x(t), with xi(t) as an implicit function governed by Bob's ODE
    ode_x = Eq(m * x.diff(t, t), -Omega**2 * x + Omega**2 * xi)

    # Solve the coupled system of ODEs
    solutions = dsolve([ode_x, ode_xi])

    # Print the solutions
    for sol in solutions:
        print(latex(sol).replace("frac", "dfrac"))

def ODE_IVP_LAGRANGIAN_FORMULATION():
    r'''
    Attempts to solve the ODE:
    
        \ddot{\theta} + \frac{g}{l} \sin (\theta) =  \frac{ A \omega^2}{l} \left( \cos (\theta) \cos(\omega t) \right)
        
    '''
    # Define symbols and functions
    t, g, l, A, omega = symbols('t g l A omega', positive=True)
    theta = Function('theta')(t)

    # Define the nonlinear ODE
    ode = Eq(theta.diff(t, t) + (g / l) * sin(theta), (A * omega**2 / l) * cos(theta) * cos(omega * t))

    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = dsolve(ode, theta)
    except Exception as e:
        print(e)
    print(latex(solution))
    
def ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR():
    r'''
    Solves the ODE:
    
        \ddot{\theta} + \frac{g}{l} \theta =  \frac{ A \omega^2}{l} \cos(\omega t) 
    '''
    # Define variables and functions
    t, g, l, A, omega = symbols('t g l A omega', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        Derivative(theta, t, t) + (g / l) * (theta) - (A * omega / l) * (1) * omega * cos(omega * t),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = dsolve(ode, theta)
    except Exception as e:
        print(e)
    print(latex(solution))

def ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_1ST_PI():
    r'''
    Solves the ODE:
        \ddot{\theta} + \frac{g}{l}(\pi - \theta) =  -\frac{ A \omega^2}{l}\cos(\omega t)
    '''
    # Define variables and functions
    t, l, g, A, omega, theta_0, theta_dot_0 = symbols('t l g A omega, theta_0, theta_dot_0', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        Derivative(theta, t, t) + (g / l) * (pi - theta) + ((A * omega * omega) / l) * cos(omega * t),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = dsolve(ode, theta)#, ics={theta.subs(t,0): theta_0, diff(theta, t).subs(t,0): theta_dot_0})
    except Exception as e:
        print(e)
    print(latex(solution))

def ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_2ND_PI():
    r'''
    Attempts to solve the ODE:
        \ddot{\theta} + \frac{g}{l}\left((\pi - \theta) + \dfrac{(\theta - \pi)^3}{6}\right) =  \frac{ A \omega^2}{l}\left(-1 + \dfrac{(x-\pi)^2}{2}\right)\cos(\omega t)
    '''
    # Define variables and functions
    t, l, g, A, omega = symbols('t l g A omega', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        Derivative(theta, t, t) + (g / l) * ((pi - theta) + (((theta - pi)**3) / 6) ) - ((A * omega * omega) / l) * (-1 + ((theta - pi)**2) / 2) * cos(omega * t),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = simplify(dsolve(ode, theta))
    except Exception as e:
        print(e)
    print(latex(solution))


def ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_1ST_PI_VERTICAL():
    r'''
    Solves the ODE:
        \ddot{\theta} + \frac{g}{l} (\pi - \theta) = -\frac{A \omega^2}{l} \left( (\pi - \theta) \cos(\omega t) \right)
    '''
    # Define variables and functions
    t, l, g, A, omega, theta_0, theta_dot_0 = symbols('t l g A omega, theta_0, theta_dot_0', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        Derivative(theta, t, t) + (g / l) * (pi - theta) + ((A * omega * omega) / l) * (pi - theta) * cos(omega * t),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = dsolve(ode, theta)#, ics={theta.subs(t,0): theta_0, diff(theta, t).subs(t,0): theta_dot_0})
    except Exception as e:
        print(e)
    print(latex(solution))

def ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_2ND_PI_VERTICAL():
    r'''
    Attempts to solve the ODE:
        \ddot{\theta} + \frac{g}{l} \left((\pi - \theta) + \dfrac{(\theta - \pi)^3}{6}\right) = -\frac{A \omega^2}{l} \left( \left((\pi - \theta) + \dfrac{(\theta - \pi)^3}{6}\right) \cos(\omega t) \right)
    '''
    # Define variables and functions
    t, l, g, A, omega = symbols('t l g A omega', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        Derivative(theta, t, t) + (g / l) * ((pi - theta) + (((theta - pi)**3) / 6) ) + ((A * omega * omega) / l) * ((pi - theta) + (((theta - pi)**3) / 6) ) * cos(omega * t),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = simplify(dsolve(ode, theta))
    except Exception as e:
        print(e)
    print(latex(solution))

def ODE_IVP_Jan_22_MAT_638():
    r'''
    Attempts to solve the ODE:
        l\ddot{\theta} + g\sin(\theta) = 0
    '''
    # Define variables and functions
    t, l, g, A, omega = symbols('t l g A omega', positive=True)
    theta = Function('theta')(t)

    # Define the ODE
    ode = Eq(
        l*Derivative(theta, t, t) + (g * sin(theta)),
        0
    )
    # Attempt to solve the ODE symbolically
    solution = None
    try:
        solution = simplify(dsolve(ode, theta))
    except Exception as e:
        print(e)
    print(latex(solution))


#ODE_IVP()
#ODE_IVP_PLUS_SOLITON()
#ODE_IVP_LAGRANGIAN_FORMULATION()
#ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR()
#ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_1ST_PI_VERTICAL()
#ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_2ND_PI()
#ODE_IVP_LAGRANGIAN_FORMULATION_TAYLOR_2ND_PI_VERTICAL()
ODE_IVP_Jan_22_MAT_638()
