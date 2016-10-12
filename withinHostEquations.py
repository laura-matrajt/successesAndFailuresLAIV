from __future__ import division
import sys
import numpy as np
import colorsys
from scipy import sparse
from scipy.integrate import odeint, simps
from matplotlib import pyplot as plt
from hillFunction import hillFunction

def influenzaWithinHostEquations(y,t, params, distanceParams):
    """

    :param t: time
    :param y: the vector of variables, with the variables defined as:
    T, I, V, B_O, A_O, B_V, A_V, T_E, T_M, T_R = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
    T = target cells
    I = infected target cells
    V = Virus
    B_O = B-cells for old strain
    A_O = antibodies for old strain
    B_V = B-cells for vaccine strain
    A_V = antibodies for vaccine strain
    T_E = expanding T- cells
    T_M = memory T-cells
    T_R = resident T-cells

    :param params: the parameters of the model:
    alpha = rate of apoptosis of expanding T-cells
    beta = virus infectivity
    c = Rate of virus clearance
    delta = infected-cell lifespan
    d_A = rate of antibody decay
    d_R = rate of resident T-cell decay
    kappa = rate of production of antibodies
    k_A = rate of killing of virus by antibodies
    k_R = rate of killing of infected cells by resident T-cells
    mu = rate of migration to site of infection
    p = Virus production per cell
    phi = Number of infected cells for half-max T-cell production
    phi_A = Virus for half-max B-cell proliferation
    r = rate of conversion from T_E to T_M
    rho = T cell proliferation rate
    sigma = rate of B-cell production

    :param distanceParams: a vector with four entries, describing the following:
    distValueO, distValueV, distValueB_O, distValueB_V
    distValue0 = distance from old strain to current strain V for antibody killing of virus
    distValueV = distance from vaccine strain to current strain V for antibody killing of virus
    distValue0 = distance from old strain to current strain V for B-cell production
    distValue0 = distance from vaccine strain to current strain V for B-cell production
    :return:
    """



    [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma] = params

    [distValueO, distValueV, distValueB_O, distValueB_V] = distanceParams

    T, I, V, B_O, A_O, B_V, A_V, T_E, T_M, T_R = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]

    #equations:
    dT = -beta*T*V

    dI = beta*T*V -k_R*T_R*I - delta*I

    dV = p*I - c*V - distValueO*k_A*A_O*V - distValueV*k_A*A_V*V

    dB_O = distValueB_O*(sigma*V)/(phi_A + V)

    dA_O = kappa * B_O - d_A * A_O

    dB_V = distValueB_V* (sigma * V) / (phi_A + V)

    dA_V = kappa*B_V - d_A*A_V

    dT_E = rho*T_E*(I/(phi + I)) - (alpha + r)*T_E*(1 - (I/(phi + I))) - mu*T_E

    dT_M = r*T_E*(1 - (I/(phi+I)))

    dT_R = mu * T_E  - d_R * T_R

    dydt = [dT, dI, dV, dB_O, dA_O, dB_V, dA_V, dT_E, dT_M, dT_R]
    return dydt




def influenzaWithinHostEquations2(t,y, params, distanceParams):
    """

    :param t: time
    :param y: the vector of variables, with the variables defined as:
    T, I, V, B_O, A_O, B_V, A_V, T_E, T_M, T_R = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]
    T = target cells
    I = infected target cells
    V = Virus
    B_O = B-cells for old strain
    A_O = antibodies for old strain
    B_V = B-cells for vaccine strain
    A_V = antibodies for vaccine strain
    T_E = expanding T- cells
    T_M = memory T-cells
    T_R = resident T-cells

    :param params: the parameters of the model:
    alpha = rate of apoptosis of expanding T-cells
    beta = virus infectivity
    c = Rate of virus clearance
    delta = infected-cell lifespan
    d_A = rate of antibody decay
    d_R = rate of resident T-cell decay
    kappa = rate of production of antibodies
    k_A = rate of killing of virus by antibodies
    k_R = rate of killing of infected cells by resident T-cells
    mu = rate of migration to site of infection
    p = Virus production per cell
    phi = Number of infected cells for half-max T-cell production
    phi_A = Virus for half-max B-cell proliferation
    r = rate of conversion from T_E to T_M
    rho = T cell proliferation rate
    sigma = rate of B-cell production

    :param distanceParams: a vector with four entries, describing the following:
    distValueO, distValueV, distValueB_O, distValueB_V
    distValue0 = distance from old strain to current strain V for antibody killing of virus
    distValueV = distance from vaccine strain to current strain V for antibody killing of virus
    distValue0 = distance from old strain to current strain V for B-cell production
    distValue0 = distance from vaccine strain to current strain V for B-cell production
    :return:
    """



    [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma] = params

    [distValueO, distValueV, distValueB_O, distValueB_V] = distanceParams

    T, I, V, B_O, A_O, B_V, A_V, T_E, T_M, T_R = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9]

    #equations:
    dT = -beta*T*V

    dI = beta*T*V - delta*I -k_R*T_R*I

    dV = p*I - c*V - distValueO*k_A*A_O*V - distValueV*k_A*A_V*V

    dB_O = distValueB_O*(sigma*V)/(phi_A + V)

    dA_O = kappa * B_O - d_A * A_O

    dB_V = distValueB_V* (sigma * V) / (phi_A + V)

    dA_V = kappa*B_V - d_A*A_V

    dT_E = rho*T_E*(I/(phi + I)) - (alpha + r)*T_E*(1 - (I/(phi + I))) - mu*T_E*I

    dT_M = r*T_E*(1 - (I/(phi+I)))

    dT_R = mu * T_E*I  - d_R * T_R

    dydt = [dT, dI, dV, dB_O, dA_O, dB_V, dA_V, dT_E, dT_M, dT_R]
    return dydt


def simpleEqs(t,y, params):

    T, I, V, T_E, T_R = y[0], y[1], y[2], y[3], y[4]#B_O, A_O, T_E, T_M, T_R = y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]
    [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma] = params
    dT = -beta * T * V

    dI = beta * T * V - delta * I  - k_R * T_R * I

    dV = p * I - c * V #- k_A*A_O*V

    # dB_O = (sigma*V)/(phi_A + V)
    #
    # dA_O = kappa * B_O - d_A * A_O
    #
    dT_E = rho*T_E*(I/(phi + I)) - (alpha + r)*T_E*(1 - (I/(phi + I))) - mu*T_E
    #
    # dT_M = r*T_E*(1 - (I/(phi+I)))
    #
    dT_R = mu * T_E  - d_R * T_R


    dydt = [dT, dI, dV, dT_E, dT_R]#, dB_O, dA_O, dT_E, dT_M, dT_R]
    return dydt

def influenzaWithinHostEquations3(y,t, params, distanceParams):
    """
    This function is a simplified version of the function influenzaWithinHostEquations. It does not have any T-cell or
    antibody compartments. In addition, the target cells equation includes a term for an innate immune response. Here,
    type I interferon causes susceptible cells to become resistant to infection.


    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below:

    :param distanceParams: vector of 2 variables measuring the antigenic distance between the B-cells (old and vaccine)
     and the virus.
    :return: a vector of the rate of change of y over the time step delta_t
    """

    [b, beta, c, delta, k, k_B, p, phi, sigma, tau] = params

    [distValueB_O, distValueB_V] = distanceParams

    T, I, V, B_O, B_V = y[0], y[1], y[2], y[3], y[4]

    #equations:
    dT = -beta*T*V - k*T*np.exp(-b*tau/2)*I(t-tau)

    dI = beta*T*V - delta*I

    dV = p*I - c*V - distValueB_O*k_B*B_O*V - distValueB_V*k_B*B_V*V

    dB_O = distValueB_O*B_O*(sigma*V)/(phi + V)

    dB_V = distValueB_V*B_V* (sigma * V) / (phi + V)

    dydt = [dT, dI, dV, dB_O, dB_V]

    return dydt
