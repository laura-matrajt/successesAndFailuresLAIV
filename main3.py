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

mylabels = ["T", "I", "V", "B_Old", "A_Old", "B_Vac", "A_Vac", "T_E", "T_M", "T_R"]
mytitles = ['Target cells', 'Infected cells', 'Viral load', 'B-cells, old strain', 'Antibodies, old strain',
            'B-cells, new strain', 'Antibodies, new strain', 'T-cells, effector', 'T-cells, memory', 'T-cells, resident']
# mylabels = ["T", "I", "V", "B_Old", "A_Old", "T_E", "T_M", "T_R"]
for tvals in range(5):
    k_R = 0.0002*tvals
    fig1 = plt.figure(tvals, figsize=(20, 10))
    plt.subplots_adjust(left=0.05, right=0.95, hspace=0.28, top=0.93, bottom=0.05)
    # if jvals == 0:
        # fig1.text(0.15, 0.97, 'First infection', fontweight='bold', fontsize=16)
        # fig1.text(0.65, 0.97, 'Second infection', fontweight='bold', fontsize=16)
    # fig1.text(0.36, 0.98, "Diff. strain is %02d" % f_vaccine + '% different, k_R = ' + str(k_R), fontweight='bold',
    #           fontsize=16)
    fig1.text(0.02, 0.95, "A) Target cells", fontweight='bold', fontsize=16)
    fig1.text(0.02, 0.63, "B) Infected cells", fontweight='bold', fontsize=16)
    fig1.text(0.02, 0.32, "C) Viral load", fontweight='bold', fontsize=16)
    for jvals in range(5):
        # Parameter values
        alpha = 0.4
        beta = 10 ** (-5)
        c = 3.0
        delta = 1
        d_A = 0.1
        d_R = 0.1
        kappa = 0.1
        k_A = 10**(-2+jvals)
        print k_A

        mu = 1.2
        p = 0.04
        phi = 50  # need to check for this as in the frontiers paper this was antigen, now it is infected cells
        phi_A = 100
        r = 0.07
        rho = 2.15
        sigma = 1

        params = [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma]

        paramsEx = [alpha, beta, c, delta, d_A, d_R, kappa, k_A, k_R, mu, p, phi, phi_A, r, rho, sigma]

        # run the model first with an "old virus" to get immunity
        # distance parameters
        f_old = 0.0
        f_vaccine = 20.0
        distValueB_O = hillFunction(f_old, 27, 5)
        distValueB_V = hillFunction(f_vaccine, 27, 5)
        print hillFunction(f_vaccine, 27, 5)
        distValueO = hillFunction(f_old, 27, 5)
        distValueV = hillFunction(f_vaccine, 27, 5)

        distanceParams = [distValueO, distValueV, distValueB_O, distValueB_V]
        # print distanceParams
        tstart = 0
        tend = 30
        delta_t = 0.5

        num_steps = np.floor((tend - tstart) / delta_t) + 1
        tspan0 = np.linspace(tstart, tend, num_steps)

        # initial conditions
        T0 = 4 * 10 ** 8
        I0 = 0
        V0 = 1  # 10**4
        B_O0 = 1
        A_O0 = 1
        B_V0 = 1
        A_V0 = 1
        T_E0 = 1
        T_M0 = 0
        T_R0 = 0

        y0long = [T0, I0, V0, B_O0, A_O0, B_V0, A_V0, T_E0, T_M0, T_R0]
        y0 = [T0, I0, V0, B_O0, A_O0, T_E0, T_M0, T_R0]
        soln = odeint(influenzaWithinHostEquations, y0long, tspan0, args=(params, distanceParams))

        ##################### Infection with a virus that is exactly like the old strain:
        # initial conditions
        T0new = 4 * 10 ** 8
        I0new = 0
        V0new = 1  # 10**4
        B_O0new = soln[-1, 3]
        A_O0new = soln[-1, 4]
        B_V0new = soln[-1, 5]
        A_V0new = soln[-1, 6]
        T_E0new = soln[-1, 7] + soln[-1, 8]
        # print 'T_E0new', T_E0new
        T_M0new = 0
        T_R0new = soln[-1, 9]
        # print 'TR0new', T_R0new

        y0new = [T0new, I0new, V0new, B_O0new, A_O0new, B_V0new, A_V0new, T_E0new, T_M0new, T_R0new]

        # new parameters:
        # distance parameters
        f_old = 0.0
        f_vaccine = 0.0
        distValueB_O = hillFunction(f_old, 27, 5)
        distValueB_V = hillFunction(f_vaccine, 27, 5)
        distValueO = hillFunction(f_old, 27, 5)
        distValueV = hillFunction(f_vaccine, 27, 5)

        distanceParamsNew = [distValueO, distValueV, distValueB_O, distValueB_V]
        print 'distanceParamsNewOldStrain', distanceParamsNew
        tstart = 50
        tend = 80
        delta_t = 0.5

        num_steps = np.floor((tend - tstart) / delta_t) + 1
        tspan1 = np.linspace(tstart, tend, num_steps)

        soln1 = odeint(influenzaWithinHostEquations, y0new, tspan1, args=(params, distanceParamsNew))

        ########### Infection with a virus that is  different from the old strain: ##########

        # new parameters:
        # distance parameters
        f_old = 0.0
        f_vaccine = 30.0
        distValueB_O = hillFunction(f_old, 27, 5)
        distValueB_V = hillFunction(f_vaccine, 27, 5)
        distValueO = hillFunction(f_old, 27, 5)
        distValueV = hillFunction(f_vaccine, 27, 5)

        distanceParamsNewVacc = [distValueO, distValueV, distValueB_O, distValueB_V]
        print 'distanceParamsNewVacc', distanceParamsNewVacc
        soln2 = odeint(influenzaWithinHostEquations, y0new, tspan1, args=(params, distanceParamsNewVacc))



        # ############ Plotting   ############
        ############ Plotting   ############
        # fig1 = plt.figure(1, figsize=(20,10))
        # plt.subplots_adjust(left=0.05, right=0.95, hspace=0.28, top=0.93, bottom=0.05)
        if jvals == 0:
        #     # fig1.text(0.15, 0.97, 'First infection', fontweight='bold', fontsize=16)
        #     # fig1.text(0.65, 0.97, 'Second infection', fontweight='bold', fontsize=16)
            fig1.text(0.36, 0.98, "Diff. strain is %02d" % f_vaccine + '% different, k_R = '+ str(k_R), fontweight='bold', fontsize=16)
        #     fig1.text(0.02, 0.95, "A) Target cells", fontweight='bold', fontsize=16)
        #     fig1.text(0.02, 0.63, "B) Infected cells", fontweight='bold', fontsize=16)
        #     fig1.text(0.02, 0.32, "C) Viral load", fontweight='bold', fontsize=16)

        # for ivals in range(1):
        #     plt.subplot(3, 6, ivals * 6 + 1)
        #     plt.plot(tspan0, np.log10(soln[:, ivals] + 1 * np.ones(np.size(soln[:, ivals]))), label='k_A=' + str(k_A),
        #         color=mycolors[jvals], linewidth=2)
        #
        #
        #     plt.subplot(3, 6, ivals * 6 + jvals + 2)
        #     # plt.plot(tspan1, np.log10(soln1[:, ivals] + 1 * np.ones(np.size(soln1[:, ivals]))), label=  # mylabels[ivals] +
        #     #          ' same virus', color=mycolors[jvals], linewidth=2)
        #
        #     plt.plot(tspan1, np.log10(soln2[:, ivals] + 1 * np.ones(np.size(soln2[:, ivals]))), '--',
        #              label=  # mylabels[ivals] +
        #              ' diff. virus', color=mycolors[jvals], linewidth= 2)


        for ivals in range(3):
            plotvals0 = ivals * 6 + 1
            plt.subplot(3, 6, plotvals0)
            plt.plot(tspan0, np.log10(soln[:, ivals] + 1 * np.ones(np.size(soln[:, ivals]))), label='k_A='+ str(k_A),
                     color=mycolors[jvals], linewidth=2)
            if ivals != 2:
                plt.ylim([0, 9])
            else:
                plt.ylim([0, 7])
            # if jvals == 0:
            plt.legend(loc=4)
            plt.ylabel("Concentration", fontweight='bold', fontsize=12)
            if plotvals0 == 13:
                plt.xlabel('Days', fontweight='bold', fontsize=12)

            # plt.title(mytitles[ivals], fontweight='bold', fontsize=14)

            # if jvals !=0:
            plotvals = ivals * 6 + jvals+2
            plt.subplot(3, 6, plotvals)
            plt.plot(tspan1, np.log10(soln1[:, ivals] + 1 * np.ones(np.size(soln1[:, ivals]))), label=  # mylabels[ivals] +
            ' same virus', color=mycolors[jvals], linewidth=2)
            plt.plot(tspan1, np.log10(soln2[:, ivals] + 1 * np.ones(np.size(soln2[:, ivals]))), '--',
                     label=  # mylabels[ivals] +
                     ' diff. virus', color=mycolors[jvals], linewidth=2)
            if ivals != 2:
                plt.ylim([0, 9])
            else:
                plt.ylim([0, 7])
            plt.ylabel("Concentration", fontweight='bold', fontsize=12)
            # if jvals == 0:
            if plotvals in range(2, 7):
                plt.legend(loc=4)
            # plt.title(mytitles[ivals], fontweight='bold', fontsize=14)
            plt.title('k_A='+ str(k_A), fontweight='bold', fontsize=14)
            # plt.title('2nd infection', fontweight='bold', fontsize=16)
            if plotvals in range(13, 19):
                plt.xlabel('Days', fontweight='bold', fontsize=14)


        # fig1 = plt.figure(2)
        # plt.subplots_adjust(left=0.08, right=0.95, hspace=0.28, top=0.93, bottom=0.05)
        # if jvals == 0:
        #     fig1.text(0.15, 0.97, 'First infection', fontweight='bold', fontsize=16)
        #     fig1.text(0.65, 0.97, 'Second infection', fontweight='bold', fontsize=16)
        # for ivals in range(3):
        #     plt.subplot(3, 2, ivals*2 + 1)
        #     plt.plot(tspan0, np.log10(soln[:, ivals] + 1 * np.ones(np.size(soln[:, ivals]))), label=mylabels[ivals],
        #              color=mycolors[jvals], linewidth=2)
        #     if ivals != 2:
        #         plt.ylim([0, 9])
        #     else:
        #         plt.ylim([0, 7])
        #     # if jvals == 0:
        #     plt.legend()
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #     plt.title(mytitles[ivals], fontweight='bold', fontsize=14)
        #
        #
        #     plt.subplot(3, 2, ivals*2 + 2)
        #     plt.plot(tspan1, np.log10(soln1[:, ivals] + 1 * np.ones(np.size(soln1[:, ivals]))),  label=#mylabels[ivals] +
        #               ' same virus', color=mycolors[jvals], linewidth=2)
        #     plt.plot(tspan1, np.log10(soln2[:, ivals] + 1 * np.ones(np.size(soln2[:, ivals]))),'--',  label=#mylabels[ivals] +
        #              ' diff. virus', color=mycolors[jvals], linewidth=2)
        #     if ivals != 2:
        #         plt.ylim([0, 9])
        #     else:
        #         plt.ylim([0, 7])
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #     # if jvals == 0:
        #     plt.legend()
        #     plt.title(mytitles[ivals], fontweight='bold', fontsize=14)
        #     # plt.title('2nd infection', fontweight='bold', fontsize=16)


        # fig2 = plt.figure(2)
        # plt.subplots_adjust(left=0.08, right=0.95, hspace=0.25, top=0.93, bottom=0.05)
        # if jvals == 0:
        #     fig2.text(0.15, 0.95, 'First infection', fontweight='bold', fontsize=16)
        #     fig2.text(0.65, 0.95, 'Second infection', fontweight='bold', fontsize=16)
        # for kvals in range(3,7):
        #     plt.subplot(4, 2, (kvals-3)*2 + 1)
        #     plt.plot(tspan0, np.log10(soln[:, kvals] + 1 * np.ones(np.size(soln[:, kvals]))), label=mylabels[kvals],
        #              color=mycolors[jvals], linewidth=2)
        #     plt.ylim([0, 9])
        #     plt.legend()
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #
        #     plt.subplot(4, 2, (kvals-3)*2 + 2)
        #     plt.plot(tspan1, np.log10(soln1[:, kvals] + 1 * np.ones(np.size(soln1[:, kvals]))),  label=mylabels[kvals]
        #             + ' same virus', color=mycolors[jvals], linewidth=2)
        #     plt.plot(tspan1, np.log10(soln2[:, kvals] + 1 * np.ones(np.size(soln2[:, kvals]))), '--', label=mylabels[kvals]
        #             + ' diff. virus', color=mycolors[jvals], linewidth=2)
        #     plt.ylim([0, 9])
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #     if jvals == 0:
        #         plt.legend()
        #
        # fig3 = plt.figure(3)
        # plt.subplots_adjust(left=0.08, right=0.95, hspace=0.25, top=0.93, bottom=0.05)
        # if jvals == 0:
        #     fig3.text(0.15, 0.95, 'First  infection', fontweight='bold', fontsize=16)
        #     fig3.text(0.65, 0.95, 'Second infection', fontweight='bold', fontsize=16)
        # for kvals in range(7,10):
        #     plt.subplot(3, 2, (kvals-7)*2 + 1)
        #     plt.plot(tspan0, np.log10(soln[:, kvals] + 1 * np.ones(np.size(soln[:, kvals]))), label=mylabels[kvals],
        #              color=mycolors[jvals], linewidth=2)
        #     plt.ylim([0, 9])
        #     plt.legend()
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #
        #     plt.subplot(3, 2, (kvals-7)*2 + 2)
        #     plt.plot(tspan1, np.log10(soln1[:, kvals] + 1 * np.ones(np.size(soln1[:, kvals]))),  label=mylabels[kvals]
        #             + ' same virus',color=mycolors[jvals], linewidth=2)
        #     plt.plot(tspan1, np.log10(soln2[:, kvals] + 1 * np.ones(np.size(soln2[:, kvals]))), '--', label=mylabels[kvals]
        #             + ' diff. virus',color=mycolors[jvals], linewidth=2)
        #     plt.ylim([0, 9])
        #     plt.ylabel("Concentration", fontweight='bold', fontsize=12)
        #     if jvals == 0:
        #         plt.legend()
    figname = 'strain %02d' % f_vaccine + '_different' + str(k_R)
    plt.savefig('figures/'+ figname + '.pdf')
plt.show()
