from __future__ import division
import sys
import numpy as np
import colorsys
from scipy import sparse
from scipy.integrate import odeint, simps
from matplotlib import pyplot as plt
from hillFunction import hillFunction


### On March 12th, 2018, I modified the functions influenzaWithinHostEquations8, 9, 10 and 11 to remove the term (k/nIntervals)
### and to leave just k. So, no normalization by the number of clones in these equations.

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
    This model doesn't work because it is a delayed differential equation and it cannot be solved with the standard
    techniques.
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

    [beta, c, delta, k_B, p, phi, sigma] = params

    [distValueB_O, distValueB_V] = distanceParams

    T, I, V, B_O, B_V = y[0], y[1], y[2], y[3], y[4]

    #equations:
    dT = -beta*T*V #- k*T*np.exp(-b*tau/2)*I[t-tau]

    dI = beta*T*V - delta*I

    dV = p*I - c*V - distValueB_O*k_B*B_O*V - distValueB_V*k_B*B_V*V

    dB_O = distValueB_O*B_O*(sigma*V)/(phi + V)

    dB_V = distValueB_V*B_V* (sigma * V) / (phi + V)

    dydt = [dT, dI, dV, dB_O, dB_V]

    return dydt


def influenzaWithinHostEquations4(y,t, params, distanceParams):
    """
    This function will be the simplest possible model of viral and immune dynamics including only equations for the virus
    and for the B-cells against each type of virus
    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below
    r = exponential growth rate for virus
    k = killing rate of virus by B-cells
    phi = number of ivurs for half-max B-cell activation
    sigma = max activation rate for B-cells
    :param distanceParams: vector of 3 variables measuring the antigenic distance between the B-cells (old, vaccine,
    challenge) and the virus.
    :return: dydt

    the model equations are the following:
    dV/dt =rV -f(BO,V)*kV*BO -f(BV,V)*kV*BV -f(BO,V)*kV*BO
    dBO/dt = sigma*BO * V/((phi/f(B0,V)) + V)
    dBV/dt = sigma*BV * V/((phi/f(BV ,V)) + V)
    dBC/dt = sigma*BC * V/((phi/f(BC ,V)) + V)
    """
    [distValueBO, distValueBV, distValueBC] = distanceParams

    [k, phi, rate, sigma] = params
    V, BO, BV, BC = y[0], y[1], y[2], y[3]
    # V, BO = y[0], y[1]

    dV = rate*V - distValueBO * k * V * BO - distValueBV * k * V * BV - distValueBC * k * V * BC

    dBO = sigma * BO * V / ((phi / distValueBO) + V)

    dBV = sigma * BV * V / ((phi / distValueBV) + V)
    # print dBV
    dBC = sigma * BC * V / ((phi / distValueBC) + V)

    return [dV, dBO, dBV, dBC]



def influenzaWithinHostEquations5(y,t, params, distanceParams, nIntervals):
    """
    THis function is different from influenzaWithinHostEquations4 because here I will consider all the different types of
    B-cells based on the distance between them and the virus.

    This function will be the simplest possible model of viral and immune dynamics including only equations for the virus
    and for the B-cells against each type of virus
    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below
    r = exponential growth rate for virus
    k = killing rate of virus by B-cells
    phi = number of ivurs for half-max B-cell activation
    sigma = max activation rate for B-cells
    :param distanceParams: vector of 3 variables measuring the antigenic distance between the B-cells (old, vaccine,
    challenge) and the virus.
    :param nIntervals: number of intervals for the distance, that is, number of cuts between a distance of 0 and a dist
    tance of 100%
    :return: dydt

    the model equations are the following:
    dV/dt =rV -sum_{i=1}^n f(Bi,V)*kV*Bi
    for i in range(n):
    dBi/dt = sigma*Bi * V/((phi/f(Bi,V)) + V)

    """


    [k, phi, rate, sigma] = params
    dV = rate * y[0]
    dydt = []

    for ivals in range(1, nIntervals+1):

        dV = dV - distanceParams[ivals-1] * k * y[0] * y[ivals]

        dBi = sigma * y[ivals] * y[0] / ((phi / distanceParams[ivals-1]) + y[0])

        dydt.append(dBi)

    dydt.insert(0, dV)
    # print dydt
    return dydt




def influenzaWithinHostEquations5bis(y,t, params, distanceParams, nIntervals):
    """
    THis function is the same as influenzaWithinHostEquations5 but I changed the name of the parameter "rate" to "beta"
    and I changed the order in which the parameters are given. I did this so that the parameter controlling the growth of
    the virus is consistently the first one in the list of parameters. This is done so that I can plot many models at once
    in the file figuresPaperAllEquations

    This function will be the simplest possible model of viral and immune dynamics including only equations for the virus
    and for the B-cells against each type of virus
    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below
    beta = exponential growth rate for virus
    k = killing rate of virus by B-cells
    phi = number of ivurs for half-max B-cell activation
    sigma = max activation rate for B-cells
    :param distanceParams: vector of 3 variables measuring the antigenic distance between the B-cells (old, vaccine,
    challenge) and the virus.
    :param nIntervals: number of intervals for the distance, that is, number of cuts between a distance of 0 and a dist
    tance of 100%
    :return: dydt

    the model equations are the following:
    dV/dt =rV -sum_{i=1}^n f(Bi,V)*kV*Bi
    for i in range(n):
    dBi/dt = sigma*Bi * V/((phi/f(Bi,V)) + V)

    """


    [beta, k, phi, sigma] = params
    dV = beta * y[0]
    dydt = []

    for ivals in range(1, nIntervals+1):

        dV = dV - distanceParams[ivals-1] * k * y[0] * y[ivals]

        dBi = sigma * y[ivals] * y[0] / ((phi / distanceParams[ivals-1]) + y[0])

        dydt.append(dBi)

    dydt.insert(0, dV)
    # print dydt
    return dydt



def influenzaWithinHostEquationsIAV(y,t, params, distanceParams, nIntervals):
    """
    THis function will model the decay of virus and stimulation of B-cells for an inactivated vaccine

    T
    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below
    r = exponential decay rate for virus
    k = killing rate of virus by B-cells
    phi = number of ivurs for half-max B-cell activation
    sigma = max activation rate for B-cells
    :param distanceParams: vector of 3 variables measuring the antigenic distance between the B-cells (old, vaccine,
    challenge) and the virus.
    :param nIntervals: number of intervals for the distance, that is, number of cuts between a distance of 0 and a dist
    tance of 100%
    :return: dydt

    the model equations are the following:
    dV/dt =rV
    for i in range(n):
    dBi/dt = sigma*Bi * V/((phi/f(Bi,V)) + V)

    """
    print "this function is wrong!"

    [k, phi, rate, sigma] = params
    dV = rate * y[0]
    dydt = []

    for ivals in range(1, nIntervals+1):

        dBi = sigma * y[ivals] * y[0] / ((phi / distanceParams[ivals-1]) + y[0])
        dydt.append(dBi)

    dydt.insert(0, dV)
    # print dydt
    return dydt


def influenzaWithinHostEquations6(y,t, params, distanceParams, nIntervals):
    """
    THis function is different from influenzaWithinHostEquations5 because here we add a death component to the B-cells.

    This function will be the simplest possible model of viral and immune dynamics including only equations for the virus
    and for the B-cells against each type of virus
    :param y: vector of variables
    :param t: time
    :param params: parameters of the model described below
    d = death rate of B-cells.
    r = exponential growth rate for virus
    k = killing rate of virus by B-cells
    phi = number of ivurs for half-max B-cell activation
    sigma = max activation rate for B-cells
    :param distanceParams: vector of 3 variables measuring the antigenic distance between the B-cells (old, vaccine,
    challenge) and the virus.
    :param nIntervals: number of intervals for the distance, that is, number of cuts between a distance of 0 and a dist
    tance of 100%
    :return: dydt

    the model equations are the following:
    dV/dt =rV -sum_{i=1}^n f(Bi,V)*kV*Bi
    for i in range(n):
    dBi/dt = sigma*Bi * V/((phi/f(Bi,V)) + V)

    """


    [d, k, phi, rate, sigma] = params
    dV = rate * y[0]
    dydt = []

    for ivals in range(1, nIntervals+1):

        dV = dV - distanceParams[ivals-1] * k * y[0] * y[ivals]

        dBi = np.max(sigma * y[ivals] * y[0] / ((phi / distanceParams[ivals-1]) + y[0]) - d*y[ivals], 0)

        dydt.append(dBi)

    dydt.insert(0, dV)
    # print dydt
    return dydt



def influenzaWithinHostEquations7(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model without any delay
    '''

    [beta, d_I, d_V, k, p, phi, sigma] = params

    dU = -beta*y[0]*y[2]
    dI = beta*y[0]*y[2] - d_I*y[1]
    dV = p*y[1] - d_V*y[2]

    dydt = [dU, dI]

    for ivals in range(3, nIntervals + 3):
        dV = dV - distanceParams[ivals - 3] * (k/nIntervals) * y[2] * y[ivals]

        dBi = sigma * y[ivals] * y[2] / ((phi / distanceParams[ivals - 3]) + y[2])

        dydt.append(dBi)

    dydt.insert(2, dV)
    # print dydt
    return dydt


def influenzaWithinHostEquations8(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model without any delay
    '''

    [beta, d_A, d_I, d_V, k, p_B, p_I, phi, sigma] = params

    dU = -beta*y[0]*y[2]
    dI = beta*y[0]*y[2] - d_I*y[1]
    dV = p_I*y[1] - d_V*y[2]

    dydt = [dU, dI]

    for ivals in range(3, nIntervals + 3):
        dBi = sigma * y[ivals] * y[2] / ((phi / distanceParams[ivals - 3]) + y[2])
        dydt.append(dBi)

    for ivals in range(3, nIntervals + 3):
        dV = dV - distanceParams[ivals - 3] * (k ) * y[2] * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals] #- \
              #distanceParams[ivals - 3] * (k / nIntervals) * y[2] * y[ivals + nIntervals]
        dydt.append(dAi)

    dydt.insert(2, dV)
    # print dydt
    return dydt



def influenzaWithinHostEquations8_normalizeNumClones(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model without any delay
    '''

    [beta, d_A, d_I, d_V, k, p_B, p_I, phi, sigma] = params

    dU = -beta*y[0]*y[2]
    dI = beta*y[0]*y[2] - d_I*y[1]
    dV = p_I*y[1] - d_V*y[2]

    dydt = [dU, dI]

    for ivals in range(3, nIntervals + 3):
        dBi = sigma * y[ivals] * y[2] / ((phi / distanceParams[ivals - 3]) + y[2])
        dydt.append(dBi)

    for ivals in range(3, nIntervals + 3):
        dV = dV - distanceParams[ivals - 3] * (k /nIntervals) * y[2] * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals] #- \
              #distanceParams[ivals - 3] * (k / nIntervals) * y[2] * y[ivals + nIntervals]
        dydt.append(dAi)

    dydt.insert(2, dV)
    # print dydt
    return dydt



def createVariableNames(nIntervals):
    '''
    THis function will create a list of the variable names for my system of ODEs with a variable number of B-cell clones.
    This list will be used to define a dictionary to create the model to be used with the dealy differential equation
    solver.
    :param nIntervals: number of B-cell clones
    :return: a list with the following items (this function will be used with a model of uninfected, infected cells, virus
    and B-cells)
    list = ['U', 'I', 'V', 'B1', ... 'Bn']
    where n = nIntervals
    '''

    mylist = []
    for ivals in range(nIntervals):
        tmp = 'B' + str(ivals)
        mylist.append(tmp)
    # print mylist
    return mylist


def writeInfluenzaEquations7pydelayFormat(nIntervals):
    mylist = createVariableNames(nIntervals)
    model = {
        'U': '-beta*U*V',
        'I': 'beta*U*V - d_I*I',
        'V': 'p_I*I - d_V*V'
    }
    for ivals in range(nIntervals):
        model['V'] = model['V'] + '-' + 'distanceParams'+ str(ivals) + '*(k/nIntervals)*V*' + mylist[ivals]
        model[mylist[ivals]] = 'sigma*'+mylist[ivals]+'*V(t-tau)/((phi/distanceParams'+ str(ivals) + ') + V(t-tau))'

    return model


def influenzaWithinHostEquations8_2clones(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model without any delay
    '''

    [beta, d_A, d_I, d_V, k, p_B, p_I, phi, sigma] = params

    U, I, V, B0, B1, A0, A1 = y[0], y[1], y[2],y[3], y[4], y[5], y[6]

    # dU = -beta*U*V
    # dI = beta*U*V - d_I*I
    # dV = p_I*I - d_V*V - (k/nIntervals)*V*(distanceParams[0]*A0 + distanceParams[1]*A1)
    # dB0 = sigma*y[3]*y[2]/((phi/distanceParams[0]) + y[2])
    # dB1 = sigma*y[4]*y[2]/((phi/distanceParams[1]) + y[2])
    # dA0 = p_B * y[3] - d_A * y[5] - (k / nIntervals) * distanceParams[0] * y[5]
    # dA1 = p_B * y[4] - d_A * y[6] - (k / nIntervals) * distanceParams[1] * y[6]

    dU = -beta * y[0] * y[2]
    dI = beta * y[0] * y[2] - d_I * y[1]
    dV = p_I * y[1] - d_V * y[2] - (k / nIntervals) * y[2] * (distanceParams[0] * y[5] + distanceParams[1] * y[6])
    dB0 = sigma * y[3] * y[2] / ((phi / distanceParams[0]) + y[2])
    dB1 = sigma * y[4] * y[2] / ((phi / distanceParams[1]) + y[2])
    dA0 = p_B * y[3] - d_A * y[5] - (k / nIntervals) * distanceParams[0] * y[5]
    dA1 = p_B * y[4] - d_A * y[6] - (k / nIntervals) * distanceParams[1] * y[6]

    dydt = [dU, dI, dV, dB0, dB1, dA0, dA1]
    return dydt


def createVariableNames2(nIntervals):
    '''
    THis function will create a list of the variable names for my system of ODEs with a variable number of B-cell clones
    and the same number of antibody clones.
    This list will be used to define a dictionary to create the model to be used with the dealy differential equation
    solver.
    :param nIntervals: number of B-cell clones and Antibody clones
    :return: a list with the following items (this function will be used with a model of uninfected, infected cells, virus
    and B-cells)
    list = ['U', 'I', 'V', 'B1', ... 'Bn', 'A1', ..., 'Bn']
    where n = nIntervals
    '''

    mylist = []
    #append the Bi's:
    for ivals in range(nIntervals):
        tmp = 'B' + str(ivals)
        mylist.append(tmp)

    #append the Ai's:
    for ivals in range(nIntervals):
        tmp = 'A' + str(ivals)
        mylist.append(tmp)
    # print mylist
    return mylist


def writeInfluenzaEquations9pydelayFormat(nIntervals):
    mylist = createVariableNames2(nIntervals)
    print mylist
    model = {
        'U': '-beta*U*V -k_X*U*I(t-tau)',
        'I': 'beta*U*V - d_I*I',
        'V': 'p_I*I - d_V*V'
    }
    for ivals in range(nIntervals):
       # model['V'] = model['V'] + '-' + 'distanceParams'+ str(ivals) + '*(k/nIntervals)*V*' + mylist[ivals]
        model[mylist[ivals]] = 'sigma*'+mylist[ivals]+'*V/((phi/distanceParams'+ str(ivals) + ') + V)'

    for ivals in range(nIntervals, 2*nIntervals):
        model['V'] = model['V'] + '-' + 'distanceParams' + str(ivals- nIntervals) + '*(k/nIntervals)*V*' + mylist[ivals]
        # print model['V']
        model[mylist[ivals]] = 'p_B*' + mylist[ivals - nIntervals] + '-' + 'd_A*'+ mylist[ivals] + '-' + \
                               'distanceParams' + str(ivals - nIntervals) + '*(k/nIntervals)*V*' + mylist[ivals]
        # print model[mylist[ivals]]
    for keys, values in model.items():
        print keys
        print values
    return model



def writeInfluenzaEquations9pydelayFormat2(nIntervals):
    mylist = createVariableNames2(nIntervals)
    print mylist
    model = {
        'U': '-beta*U*V -k_X*U*I(t-tau)',
        'I': 'beta*U*V - d_I*I',
        'V': 'p_I*I - d_V*V'
    }
    for ivals in range(nIntervals):
       # model['V'] = model['V'] + '-' + 'distanceParams'+ str(ivals) + '*(k/nIntervals)*V*' + mylist[ivals]
        model[mylist[ivals]] = 'sigma*'+mylist[ivals]+'*V/((phi/distanceParams'+ str(ivals) + ') + V)'

    for ivals in range(nIntervals, 2*nIntervals):
        model['V'] = model['V'] + '-' + 'distanceParams' + str(ivals- nIntervals) + '*(k/nIntervals)*V*' + mylist[ivals]
        # print model['V']
        model[mylist[ivals]] = 'p_B*' + mylist[ivals - nIntervals] + '-' + 'd_A*'+ mylist[ivals] + '-' + \
                               'distanceParams' + str(ivals - nIntervals) + '*(k/nIntervals)*V*' + mylist[ivals]
        # print model[mylist[ivals]]
    # for keys, values in model.items():
    #     print keys
    #     print values
    return model


def influenzaWithinHostEquations9(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model with innate immune response without any delay
    '''

    [beta, d_A, d_I, d_V, k, k_x, p_B, p_I, phi, sigma] = params

    dU = -beta*y[0]*y[2] - k_x*y[0]*y[1]
    dI = beta*y[0]*y[2] - d_I*y[1]
    dV = p_I*y[1] - d_V*y[2]

    dydt = [dU, dI]

    for ivals in range(3, nIntervals + 3):
        dBi = sigma * y[ivals] * y[2] / ((phi / distanceParams[ivals - 3]) + y[2])
        dydt.append(dBi)

    for ivals in range(3, nIntervals + 3):
        dV = dV - distanceParams[ivals - 3] * (k) * y[2] * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals] #- \
              #distanceParams[ivals - 3] * (k / nIntervals) * y[2] * y[ivals + nIntervals]
        dydt.append(dAi)

    dydt.insert(2, dV)
    # print dydt
    return dydt


def influenzaWithinHostEquations10(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model with innate immune response without any delay
    '''

    [beta, d_A, d_I, d_V, d_x, k, k_x, p_B, p_I, phi, phi_x, sigma, sigma_x] = params


    dU = -beta*y[0]*y[2] - k_x*y[0]*y[3]
    dI = beta*y[0]*y[2] - d_I*y[1]
    dV = p_I*y[1] - d_V*y[2]
    dX = sigma_x*(100 - y[3])*(y[2]/(phi_x + y[2])) - d_x*y[3]

    dydt = [dU, dI, dX]

    for ivals in range(4, nIntervals + 4):
        dBi = sigma * y[ivals] * y[2] / ((phi / distanceParams[ivals - 4]) + y[2])
        dydt.append(dBi)

    for ivals in range(4, nIntervals + 4):
        dV = dV - distanceParams[ivals - 4] * (k) * y[2] * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals] #- \
              #distanceParams[ivals - 4] * (k / nIntervals) * y[2] * y[ivals + nIntervals]
        dydt.append(dAi)

    dydt.insert(2, dV)
    # print dydt
    return dydt



def influenzaWithinHostEquations10_oneClone(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model with innate immune response without any delay
    '''

    [beta, d_A, d_I, d_V, d_x, k, k_x, p_B, p_I, phi, phi_x, sigma, sigma_x] = params

    U = y[0]        # uninfected cells
    I = y[1]        # infected cells
    V = y[2]        # virus
    X = y[3]        # innate immunity
    B = y[4]        # B cells
    A = y[5]    # antibodies

    # uninfected cells
    dU = -beta*y[0]*y[2] - k_x*y[0]*y[3]
    # infected cells
    dI = beta * U * V - d_I * I
    # virus
    dV = p_I * I - d_V * V - k * V * A
    # innate immunity
    dX = sigma_x * (100 - X) * V / (phi_x + V) - d_x * X
    # B cells
    dB = sigma * B * V / (phi + V)
    # Antibody
    dA = p_B * B - d_A * A

    dydt = [dU, dI, dV, dX, dB, dA]

    return dydt



def influenzaWithinHostEquations11(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model with innate immune response without any delay and without uninfected cells
    I = y[0]
    V = y[1]
    X = y[2]
    B-cells = y[3] : y[3 + nIntervals]
    Antibodies = y[3 + nIntervals + 1] : y[3 + 2* nIntervals + 1]
    '''

    [beta, c, d_A, d_V, d_X, k, k_X, p_B, phi_B, phi_x, sigma_B, sigma_X] = params

    dI = beta*y[0] - k_X*y[0]*y[2]
    dV = k_X*y[0]*y[2] - d_V*y[1]
    dX = sigma_X*(100 - y[2])*(y[0]/(phi_x + y[0])) - d_X*y[2]

    dydt = [dV, dX]

    for ivals in range(3, nIntervals + 3):
        dBi = sigma_B * y[ivals] * (1 - y[ivals]/(c + y[ivals])) * y[1] / ((phi_B / distanceParams[ivals - 3]) + y[1])
        dydt.append(dBi)

    for ivals in range(3, nIntervals + 3):
        dI = dI - distanceParams[ivals - 3] * (k ) * y[0] * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals]

        dydt.append(dAi)

    dydt.insert(0, dI)
    # print dydt
    return dydt


def influenzaWithinHostEquations12(y,t, params, distanceParams, nIntervals):
    '''
    This is a target-cell limitation model with innate immune response without any delay and without uninfected cells
    I = y[0]
    V = y[1]
    X = y[2]
    B-cells = y[3] : y[3 + nIntervals]
    Antibodies = y[3 + nIntervals + 1] : y[3 + 2* nIntervals + 1]
    '''
    I = y[0]
    V = y[1]
    X = y[2]
    [beta, c, d_A, d_V, d_X, k, k_X, p_B, phi_B, phi_X, s_B, s_X] = params

    dI = beta*I - k_X*I*X
    dV = k_X*I*X - d_V*V
    dX = s_X*(100 - X)*(I/(phi_X + I)) - d_X*X

    dydt = [dV, dX]

    for ivals in range(3, nIntervals + 3):
        dBi = s_B * y[ivals] * (1 - y[ivals]/c) * y[1] / ((phi_B / distanceParams[ivals - 3]) + y[1])
        dydt.append(dBi)

    for ivals in range(3, nIntervals + 3):
        dI = dI - distanceParams[ivals - 3] * (k) * I * y[ivals + nIntervals]
        dAi = p_B * y[ivals] - d_A * y[ivals + nIntervals]

        dydt.append(dAi)

    dydt.insert(0, dI)
    # print dydt
    return dydt


if __name__=='__main__':
    # Y = [1,2,3]
    # mylist = createVariableNames(3)
    # mydic = {'V': '5*V'}
    # for ivals in range(3):
    #     mydic[mylist[ivals]] = Y[ivals]
    #     mydic['V'] = mydic['V'] + '+' + str(ivals)
    # print mydic

    print writeInfluenzaEquations9pydelayFormat(2)