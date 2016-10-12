from __future__ import division
import sys
import numpy as np
import colorsys
from scipy import sparse
from scipy.integrate import odeint, ode
from matplotlib import pyplot as plt
from hillFunction import hillFunction
from withinHostEquations import influenzaWithinHostEquations, influenzaWithinHostEquations2, simpleEqs

#define colors for plotting
Ncol = 10
HSV_tuples = [(x * 1.0 / Ncol, 0.5, 0.5) for x in range(Ncol)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
mycolors = ['#FF33DE', 'b', 'g', 'r', 'm', 'c', 'k', '#FFC300','#900C3F', '#008080','#808080']

mylabels = ["T", "I", "V", "B_O", "A_O", "B_V", "A_V", "T_E", "T_M", "T_R"]
mylabels = ["T", "I", "V", "B_O", "A_O", "T_E", "T_M", "T_R"]

#Parameter values
alpha = 0.4
beta = 10**(-6)
c = 3.0
delta = 1
d_A = 0.1
d_R = 0.1
kappa = 0.1
k_A = 0.
k_R = 0.000
mu = 1.2
p = 0.04
phi = 50 #need to check for this as in the frontiers paper this was antigen, now it is infected cells
phi_A = 100
r = 0.07
rho = 2.15
sigma = 1

params = [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma]

paramsEx = [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma]

#run the model first with an "old virus" to get immunity
#distance parameters
f_old = 0.0
f_vaccine = 100.0
distValueB_O = hillFunction(f_old, 27, 5)
distValueB_V = hillFunction(f_vaccine, 27, 5)
print hillFunction(f_vaccine, 27, 5)
distValueO = hillFunction(f_old, 27, 5)
distValueV = hillFunction(f_vaccine, 27, 5)

distanceParams = [distValueO, distValueV, distValueB_O, distValueB_V]
print distanceParams
tstart = 0
tend = 600
delta_t = 1

num_steps = np.floor((tend - tstart)/delta_t) + 1
tspan0 = np.linspace(tstart, tend, num_steps)

#initial conditions
T0 = 4*10**8
I0 = 0
V0 = 1 #10**4
B_O0 = 1
A_O0 = 1
B_V0 = 0
A_V0 = 0
T_E0 = 1
T_M0 = 0
T_R0 = 0

y0long = [T0, I0, V0, B_O0, A_O0, B_V0, A_V0, T_E0, T_M0, T_R0]
y0 = [T0, I0, V0, T_E0, T_R0]#, B_O0, A_O0, T_E0, T_M0, T_R0]
soln = odeint(influenzaWithinHostEquations, y0long, tspan0, args=(params, distanceParams))
# print soln[-100:-50, :3]

#alternative solver
# BDF method suited to stiff systems of ODEs
modelout = np.zeros((num_steps, 5))
modelout[0,:] = y0
# model = ode(influenzaWithinHostEquations2)
model = ode(simpleEqs)
model.set_integrator('vode',nsteps=500,method='bdf', atol=10**(-15))
model.set_initial_value(y0,tstart)
model.set_f_params(paramsEx)#,distanceParams)

#start the counter:
k = 1
t = np.zeros((num_steps, 1))
#run the model and store the results:
while model.successful() and k < num_steps:
   model.integrate(model.t + delta_t)

   t[k] = model.t
   modelout[k, :] = model.y
   # if (modelout[k, 0] < -10**(-8)) | (modelout[k, 1] < -10**(-8)) | (modelout[k, 2] < -10**(-8)):
   #     print "problem"
   #     print k
   #     print modelout[k,:]
   #     break

   k += 1

print modelout[:, :3]

#Infection with a virus that is somewhat similar to the old strain:
# y0new = np.copy(soln[-1,:])
#
# y0new[2] = 1
# print y0new


plt.figure(1)
plt.subplot(1, 2, 1)
for ivals in range(1):
    # plt.figure(ivals + 1)
    # plt.plot(tspan0, soln[:, ivals], label=mylabels[ivals],
    #          color=mycolors[ivals], linewidth=2)
    plt.plot(tspan0, np.log10(modelout[:, ivals]), label=mylabels[ivals],
             color=mycolors[ivals], linewidth=2)
    # plt.plot(tspan0, soln[:, ivals], '--', label=mylabels[ivals],
    #          color=mycolors[ivals], linewidth=2)
    # plt.plot(tspan0, np.log10(soln[:,ivals]+ np.ones(np.size(soln[:, ivals]))),
    #          label = mylabels[ivals], color=mycolors[ivals],linewidth=2)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(tspan0, np.log10(modelout[:, 2] + 10*np.ones(np.size(modelout[:, 2]))), linewidth=2)
# plt.plot(tspan0, np.log10(soln[:, 2] + 10*np.ones(np.size(soln[:, 2]))), '--', linewidth=2)


# plt.figure(2)
# plt.subplot(1,2,1)
# # plt.plot(tspan0, np.log10(modelout[:, 3] + 10*np.ones(np.size(modelout[:, 4]))), label=mylabels[3], color='r',linewidth=2)
# # plt.plot(tspan0, np.log10(modelout[:, 4] + 10*np.ones(np.size(modelout[:, 6]))), label=mylabels[4], color='b',linewidth=2)
# plt.plot(tspan0, np.log10(soln[:, 3] + 10*np.ones(np.size(soln[:, 4]))), '--',label=mylabels[3], color='r',linewidth=2)
# plt.plot(tspan0, np.log10(soln[:, 4] + 10*np.ones(np.size(soln[:, 6]))), '--',label=mylabels[4], color='b',linewidth=2)
# plt.legend()
#
# plt.subplot(1,2,2)
# for ivals in range(5,8):
#     # plt.figure(ivals + 1)
#     # plt.plot(tspan0, soln[:, ivals], label=mylabels[ivals],
#     #          color=mycolors[ivals], linewidth=2)
#     # plt.plot(tspan0, modelout[:, ivals], label=mylabels[ivals],
#     #          color=mycolors[ivals], linewidth=2)
#     plt.plot(tspan0, soln[:, ivals+2], '--', label=mylabels[ivals],
#              color=mycolors[ivals], linewidth=2)
# plt.legend()
plt.show()
