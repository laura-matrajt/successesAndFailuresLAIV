from __future__ import division
import sys
import numpy as np
import csv
import seaborn as sns
import colorsys
import matplotlib
import timeit
from scipy import sparse
from scipy.integrate import odeint, ode, simps, trapz
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from hillFunction import hillFunction, hillFunctionCentered
from createMatrices import createDistanceMatrix, relativeDistance
from withinHostEquations import influenzaWithinHostEquations5bis, influenzaWithinHostEquations8, \
    influenzaWithinHostEquations9, influenzaWithinHostEquations10, influenzaWithinHostEquations12
from textwrap import wrap
from matplotlib.colors import ListedColormap
from brokenaxesUpdated import brokenaxes
import matplotlib.gridspec as gridspec

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

#define colors for plotting
Ncol = 24
current_palette_Ncol = sns.color_palette("hls", Ncol)
twoColorPalette = sns.color_palette("RdBu_r", 2)

fourColorPalette = sns.color_palette("Paired", 4)
fiveColorPalette = sns.color_palette("husl", 5)
twoColorPalette1 = [fourColorPalette[1], fourColorPalette[3]]#sns.color_palette("Paired", 2)

contourPalette = sns.color_palette(sns.color_palette("coolwarm", 25))
contourPalette1 = sns.color_palette("BrBG", 25)
# contourPalette = sns.color_palette(sns.light_palette("sky blue", 25, input="xkcd"))

# print current_palette_Ncol
HSV_tuples = [(x * 1.0 / Ncol, 1,1) for x in range(Ncol)]
RGB_tuples = current_palette_Ncol#map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
# print RGB_tuples
# cp=plt.get_cmap('viridis')
color_sequence = ['#1f77b4', 'g', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5', '#aec7e8']
mymarkers = ['8', 'v', '^', 'o', 's']

# sns.palplot(sns.husl_palette(Ncol, l=.3, s=.8))



def viralDynamicsPrimaryInfection2PieCharts(myfunName, antibodies, initCond, Bindex, mylabels, params, paramsVac, nIntervals, hillParam1,
                                  hillParam2, myfig, mymodelname):
    """
    Produces a figure with 3 panels: left panel is a plot of the dynamics of a primary infection, where the virus is at
    antigenic distance 1 from pre-existing immunity.
    Right panels: Pie charts with the distribution of B-cell clones before (top) and after (bottom) infection
    NOTE: in this function the B-cell equations and
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.

    :param myfunName: function describing the ODEs to use
    :param antibodies: receives 0 or 1, 0 if model doesn't include antibodies, 1 if it does
    :param initCond: Initial conditions for that particular set of ODEs
    :param Bindex: index of the first B-cell equation
    :param mylabels: labels to be used in the dynamics plot, depending on the model, it would have different labels
    (V, I, etc)
    :param params: parameters to use for the wild-type virus
    :param paramsVac: parameters to use for an attenuated virus (the parameters themselves reflect the attenuation)
    :param nIntervals: number of clones
    :param hillParam1: parameter 1 for Hill equation
    :param hillParam2: parameter 2 for Hill equation
    :param myfig: figure number
    :param mymodelname: name to be put as a title for the figure
    :return: a figure handle
    """
    ###### Panel A: Hill function
    myclones = ['Clone ' + str(jvals) for jvals in range(1, nIntervals+1)]
    Vmax = -1
    mycolors = ['b', 'g']
    x = np.linspace(0, 100, 100)

    myk = [10, 30, 50]
    mynvals = [5, 6, 7, 8]

    ######## Plotting   #######
    fig = plt.figure(1, figsize=(16, 12))
    fig.text(0.07, 0.92, 'A) Modeled dynamics of a primary infection', fontsize=12, fontweight="bold")
    fig.text(0.57, 0.92, 'B) B-cell distribution before infection', fontsize=12, fontweight="bold")
    fig.text(0.57, 0.47, 'C) B-cell distribution after infection', fontsize=12, fontweight="bold")
    # fig.text(0.54, 0.50, 'D)', fontsize=12, fontweight="bold")

    plt.subplots_adjust(left=0.1, right=0.9, wspace=0.1)
    # plt.subplots_adjust(left=0.05, right=0.95, wspace=0.15)
    # ax = fig.add_subplot(2, 2, 1)

    loc1 = 1
    loc2 = 1.45
    myticks = [loc1, loc2]
    mywidth = 0.25

    ####### compute the viral dynamics for a wild-type virus
    # computes the distance matrix
    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    # ####Parameters
    tstart0 = 0
    tfinal0 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    #primary infection with a wild-type virus
    soln0 = odeint(myfunName, initCond, tspan0, args=(params, distanceMat[0, :], nIntervals))

    # Computing the total number of B-cells and antibodies:

    Btotal = np.sum(soln0[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln0[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln0[:, 1]))

    relativeIncreaseB1_us = soln0[-1, Bindex:Bindex + nIntervals] - \
                         soln0[0, Bindex:Bindex + nIntervals]

    perB= createVectorPercentages(relativeIncreaseB1_us, nIntervals)

    Bb = simps(Btotal, tspan0)

    if antibodies:
        AbTotal = np.sum(soln0[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln0[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln0[:, 1]))

        relativeIncreaseAntibodies = soln0[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                         soln0[0, Bindex + nIntervals:Bindex + 2 * nIntervals]

        perAb = createVectorPercentages(relativeIncreaseAntibodies, nIntervals)
        percentageAb = 1.0*perAb/np.sum(perAb)
        percentageAb = percentageAb.tolist()
        Abb = simps(AbTotal, tspan0)
    ################ PLotting: ##############################################

    ax = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    # Plot uninfected and infected cells, as well as virus (depending on the equations some of these will not be present):

    plt.plot(tspan0, (soln0[:, 0] + 1), '--', #marker= mymarkers[0],
              color= fiveColorPalette[0], markevery=20, markersize=8,
                 linewidth=3, label=mylabels[0], alpha = 1.0)
    plt.plot(tspan0, (soln0[:, 1] + 1), ':', marker=mymarkers[2],markersize=8,
             color=fiveColorPalette[2], markevery=20,
             linewidth=3, label=mylabels[1], alpha = 1.0)
    plt.plot(tspan0, (soln0[:, 2] + 1), marker=mymarkers[2], color=sns.xkcd_rgb["pale red"], markevery=20, markersize=8,
             linewidth=3, label=mylabels[2])
    plt.plot(tspan0, Btotal[:], color=sns.xkcd_rgb["denim blue"], linewidth=3, marker=mymarkers[3], markevery=20,markersize=8,label='B-cell Total')
    if antibodies:
        plt.plot(tspan0, AbTotal[:], color=sns.xkcd_rgb["medium green"], linewidth=3, marker=mymarkers[4], markevery=20, markersize=8,label='Antibodies Total')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim((10 ** 0, 10 ** 9))
    plt.xlim((0, 15))
    plt.ylabel('Concentration', fontsize=12, fontweight='bold')
    plt.xlabel('Days post-infection', fontsize=12, fontweight='bold')
    plt.yscale('log')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='upper right', fancybox=True,
              framealpha=0.5, ncol=2, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax = plt.subplot2grid((2, 2), (0, 1))
    initPerc = [100/nIntervals for ivals in range(int(nIntervals))]


    labels3 = ['Clone '+ str(i) for i in range(1, nIntervals + 1)]
    wedges, texts = ax.pie(initPerc,  colors=RGB_tuples, startangle=0)
    # wedges = ax.pie(initPerc, colors=RGB_tuples, startangle=0, labels=labels3, autopct='%1.1f%%')
    # plt.setp(autotexts, size=8, weight="bold")
    for w in wedges:
        w.set_linewidth(1)
        w.set_edgecolor('gray')

    plt.legend(labels3, bbox_to_anchor=(1.3, 0.85),
              ncol=1, fontsize=10)
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)



    ax = plt.subplot2grid((2, 2), (1, 1))
    # def func(pct, allvals):
    #     absolute = int(pct / 100. * np.sum(allvals))
    #     return "{:.1f}%\n({:d} g)".format(pct, absolute)

    # explode = (0, 0, 0, 0.5)

    mylist = percentageAb[0:3]
    all_others = np.sum(percentageAb[3:-1])

    mylist.append(all_others)



    labels = ['Clone 1',  'Clone 2', 'Clone 3', "All other \nclones"]#'Clone 4', 'Clone 5', 'Clone 6'

    wedges, texts = ax.pie(mylist,  colors=RGB_tuples[0:4], startangle=-40, labels = labels)
    # plt.setp(autotexts, size=8, weight="bold")
    for w in wedges:
        w.set_linewidth(1)
        w.set_edgecolor('gray')
    ax.set_aspect('equal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


    return fig



def figurePreviousImmunityVaccineGenericStackedBars(myfunName, antibodies, initCond, Vindex, Bindex, params, paramsVac, nIntervals,
                                         hillParam1, hillParam2, myvaccinestrains, myfig):
    '''
    This figure will plot previous infection and vaccination for two different vaccine strains that are
    somewhat close and distant from previous immunity.
    For each of these, there are two panels:
    Right panel shows the viral load of primary infection followed by vaccination.
    Letf panel shows the total antibody concentration for each clone after primary infection (blue) and the increase
    following vaccination (green)
    :return: figure handle
    '''

    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)
    myclones = [str(jvals) for jvals in range(1, nIntervals + 1)] #+ [ 'Clone ' + str(jvals) for jvals in range(1, nIntervals + 1)]
    ##################### Primary infection: #####################################
    ####Time parameters
    tstart0 = 0
    tfinal0 = 100
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    soln0 = odeint(myfunName, initCond, tspan0, args=(params, distanceMat[0, :], nIntervals))
    Btotal = np.sum(soln0[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(
        soln0[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln0[:, 1]))

    if antibodies:
        AbTotal = np.sum(soln0[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
            soln0[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln0[:, 1]))

    ########################### Vaccination with LAIV #########################################

    vaccineSol = []
    BSol = []
    AbSol = []
    relInc = []
    relIncB = []
    for ivals in range(3):
        closestClone = myvaccinestrains[ivals]
        # print closestClone
        np.set_printoptions(precision=3)
        # print distanceMat[closestClone, :]

        initCond1 = np.copy(soln0[-1, :])
        for kvals in range(Bindex):
            initCond1[kvals] = initCond[kvals]

        # time span
        tstart1 = 20
        tfinal1 = 35
        delta_t = 0.1
        num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
        tspan1 = np.linspace(tstart1, tfinal1, num_steps)

        soln1 = odeint(myfunName, initCond1, tspan1,
                       args=(paramsVac, distanceMat[closestClone, :], nIntervals))
        vaccineSol.append(soln1)


        relativeIncreaseB0 = soln0[-1, Bindex:Bindex + nIntervals] - \
                                      soln0[0, Bindex:Bindex + nIntervals]
        relativeIncreaseB1 = soln1[-1, Bindex:Bindex + nIntervals] - \
                                      soln1[0, Bindex:Bindex + nIntervals]

        relativeIncreaseB = np.array([relativeIncreaseB0, relativeIncreaseB1]).transpose()
        relIncB.append(relativeIncreaseB)

        Btotal1 = np.sum(soln1[:, Bindex:(Bindex + nIntervals)], 1) \
              - np.sum(soln1[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1[:, 1]))
        if antibodies:
            AbTotal1 = np.sum(soln1[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) \
               - np.sum(soln1[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1[:, 1]))
            AbSol.append(AbTotal1)

            relativeIncreaseAntibodies0 = soln0[-1, Bindex+nIntervals:Bindex+2*nIntervals] - \
                                          soln0[0, Bindex+nIntervals:Bindex+2*nIntervals]
            relativeIncreaseAntibodies1 = soln1[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                          soln1[0, Bindex + nIntervals:Bindex + 2 * nIntervals]


            relativeIncreaseAntibodies = np.array([relativeIncreaseAntibodies0, relativeIncreaseAntibodies1]).transpose()
            relInc.append(relativeIncreaseAntibodies)


        BSol.append(Btotal1)



##################    Plot the previous immunity and vaccine:    #############################

    if antibodies:

        mytitles = ['Viral dynamics', 'B-cell response', 'Antibody response']
        fig = plt.figure(myfig, figsize=[15,10])
        fig.text(0.02, 0.95, 'A) Vaccination with a vaccine 10\% different from pre-existing immunity', fontsize=12, fontweight="bold")
        fig.text(0.02, 0.45, 'B) Vaccination with a vaccine 45\% different from pre-existing immunity', fontsize=12, fontweight="bold")
        # fig.text(0.02, 0.33, 'C)', fontsize=12, fontweight="bold")
        plt.subplots_adjust(left=0.06, right=0.95, wspace=0.25, hspace=0.5, top=0.93, bottom=0.1)
        loc1 = 2.5
        loc2 = tspan1[0]

        for kvals in range(2):
            ax = plt.subplot(2, 2, (2*kvals +1))

            plt.plot(tspan0[:100], (soln0[:100, Vindex] + 1), color=fourColorPalette[1], label='Primary infection', linewidth=3)
            plt.plot(tspan1[:100], (vaccineSol[kvals][:100, Vindex] + 1), color=fourColorPalette[3], markevery=3,
                     label='Vaccine', linewidth=3)#+ r'$d=$' +
                           # str(int(relativeDistance(nIntervals, myvaccinestrains[kvals], 0))) + '\%', linewidth=3)
            plt.ylim([10 ** 0, 10 ** 6])
            plt.yscale('log')
            plt.xticks([loc1, loc2])
            ax.set_xticklabels(['Primary \ninfection', 'Vaccination'], fontsize=12, fontweight='bold')
            plt.ylabel('Viral Concentration', fontsize=12, fontweight='bold')
            plt.xlabel('Time (days post infection)', fontsize=12, fontweight='bold')
            plt.legend(loc=9, fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            # ax = plt.subplot(gs[kvals, 1])
            ax = plt.subplot(2, 2, 2*kvals  + 2)
            barWidth = 0.85
            r1 = np.arange(1, nIntervals+1)
            r2 = [x + barWidth for x in r1]
            mybarlength = (r2[0] - r1[0]) / 2
            r3 = [x + mybarlength for x in r1]
            print r3
            bars1 = soln0[-1, Bindex+nIntervals: Bindex+2*nIntervals]
            bars2 = vaccineSol[kvals][-1, Bindex+nIntervals: Bindex+2*nIntervals]
            plt.bar(r1, bars1, color=fourColorPalette[1], edgecolor = "white",  width = barWidth, label='Primary infection')
            plt.bar(r1, bars2, bottom=bars1, color=fourColorPalette[3], edgecolor = "white",  width = 0.85, label='Vaccine')
            plt.xticks(r1, '',  fontsize=10)
            plt.xlabel('Clone number', fontsize=12, fontweight='bold')
            plt.ylim([10 ** 0, 10 ** 7])
            plt.yscale('log')
            plt.ylabel('Antibody Overall Concentration',fontsize=12, fontweight='bold')
            if kvals == 0 or kvals ==1:
                plt.legend(loc=1, fontsize=12)
            else:
                plt.legend(loc=1, fontsize=12)
            # plt.xticks([loc1, loc2])
            ax.set_xticklabels(myclones, fontsize=10, fontweight='bold')
            ax.spines['top'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')


    return fig



def figure4WindowsAntibodies(myfunName, antibodies, initCond, Vindex, Bindex, params, params1, params2, nIntervals,
                                 hillParam1, hillParam2, cloneVaclist, cloneChallengelist, myfig, mymodelname):
    '''
    This figure will plot four windows to represent four scenarios:
    1) vaccine is close to Previous immunity AND challenge is close to previous immunity
    2) vaccine is close to Previous immunity AND challenge is Far from previous immunitya
    3) vaccine is FAR from previous immunity and challenge is CLOSE to previous immunity
    4) vaccine is FAR from previous immunity and challenge is far from previous immunity

    The results of this simulation also depend on the distance between the vaccine and the challenge.
    For each window, we will plot three curves:

     The first one is with the reference virus, the second one with a
     vaccine strain and the third one with a challenge.
    NOTE: in this function the B-cell equations and
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.



    :param myfunName: function describing the ODEs to use
    :param antibodies: receives 0 or 1, 0 if model doesn't include antibodies, 1 if it does
    :param initCond: Initial conditions for that particular set of ODEs
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the first B-cell clone
    :param params: parameters to be used for the wild-type virus
    :param params1: parameters to be used for an attenuated virus (the parameters themselves reflect the attenuation)
    :param params2: parameters to be used with a challenge virus
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param cloneVaclist: : a list with 2 numbers representing the distance between vaccine and reference virus
    :param cloneChallengelist: a list with 2 numbers representing the distance between challenge and reference virus
    :param myfig:  figure number
    :param mymodelname: the title of the figure
    :return: a figure handle

    NOTE: because of the way we are thinking distances, once we determine the distance from vaccine to reference and
    from challenge to reference, we have determined the distance from vaccine to challenge
    '''

    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    ##################### a typical influenza infection: #####################################
    # run this bit to get previous immunity
    ####Time parameters
    tstart0 = 0
    tfinal0 = 30
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    soln0 = odeint(myfunName, initCond, tspan0, args=(params, distanceMat[0, :], nIntervals))

    Btotal = np.sum(soln0[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln0[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln0[:, 1]))

    if antibodies:
        AbTotal = np.sum(soln0[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln0[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln0[:, 1]))

    ########################### Vaccination with LAIV #########################################
    # initial conditions:
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCond[ivals]

    tstart1 = 30
    tfinal1 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1 = odeint(myfunName, initCond1, tspan1,
                   args=(params1, distanceMat[cloneVaclist[0], :], nIntervals))

    Btotal1 = np.sum(soln1[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1[:, 1]))


    soln1bis = odeint(myfunName, initCond1, tspan1,
                   args=(params1, distanceMat[cloneVaclist[1], :], nIntervals))

    Btotal1bis = np.sum(soln1bis[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1bis[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1[:, 1]))

    if antibodies:
        AbTotal1 = np.sum(soln1[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
            soln1[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1[:, 1]))
        AbTotal1bis = np.sum(soln1bis[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln1bis[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1bis[:, 1]))

    vaccineSols = [soln1, soln1bis]
    vaccineAb = [AbTotal1, AbTotal1bis]
    vaccineB = [Btotal1, Btotal1bis]


    ################### Infection with a challenge after vaccination ##################################
    #Two strains given by the user:
    closestClone2 = cloneChallengelist[0]
    closestClone2_1 = cloneChallengelist[1]

    mydistances = [closestClone2, closestClone2_1]
    myadjectives = [' close', ' far']
    #run the challenge with each of the vaccine viruses:
    challengeSolns = []
    BtotalSolns = []
    AbTotalSolns = []
    for kvals in range(2):
        initCond2 = np.copy(vaccineSols[kvals][-1, :])
        for ivals in range(Bindex):
            initCond2[ivals] = initCond[ivals]

        tstart2 = 60
        tfinal2 = 90
        delta_t = 0.1
        num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
        tspan2 = np.linspace(tstart2, tfinal2, num_steps)

        soln2 = odeint(myfunName, initCond2, tspan2,
                       args=(params2, distanceMat[closestClone2, :], nIntervals))

        soln2_1 = odeint(myfunName, initCond2, tspan2,
                         args=(params2, distanceMat[closestClone2_1, :], nIntervals))


        challenge = [soln2, soln2_1]
        BtotalCh = []
        AbTotalCh = []

        for jvals in range(2):
            BtotalTemp = np.sum(challenge[jvals][:, Bindex:(Bindex + nIntervals)], 1) - np.sum(
                challenge[jvals][0, Bindex:(Bindex + nIntervals)]) * np.ones(
                np.size(challenge[jvals][:, 1]))
            BtotalCh.append(BtotalTemp)
            if antibodies:
                AbTotalTemp = np.sum(challenge[jvals][:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
                challenge[jvals][0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(challenge[jvals][:, 1]))
                AbTotalCh.append(AbTotalTemp)
        challengeSolns.append(challenge)
        BtotalSolns.append(BtotalCh)
        AbTotalSolns.append(AbTotalCh)


    #     ################### Infection with a challenge wihtout vaccination ##################################

    mydistances = [closestClone2, closestClone2_1]
    myadjectives = [' close', ' far']

    #run the challenge with each of the vaccine viruses:
    challengeSolnsWithoutVaccine = []
    BtotalSolnsWithoutVaccine = []
    AbTotalSolnsWithoutVaccine = []
    for kvals in range(2):
        initCond2bis = np.copy(soln0[-1, :])
        for ivals in range(Bindex):
            initCond2bis[ivals] = initCond[ivals]

        # print initCond2bis

        tstart2 = 60.9
        tfinal2 = 90.9
        delta_t = 0.1
        num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
        tspan2bis = np.linspace(tstart2, tfinal2, num_steps)

        soln2 = odeint(myfunName, initCond2bis, tspan2bis,
                       args=(params2, distanceMat[closestClone2, :], nIntervals))

        soln2_1 = odeint(myfunName, initCond2bis, tspan2bis,
                         args=(params2, distanceMat[closestClone2_1, :], nIntervals))


        challenge = [soln2, soln2_1]
        BtotalCh = []
        AbTotalCh = []

        for jvals in range(2):
            BtotalTemp = np.sum(challenge[jvals][:, Bindex:(Bindex + nIntervals)], 1) - np.sum(
                challenge[jvals][0, Bindex:(Bindex + nIntervals)]) * np.ones(
                np.size(challenge[jvals][:, 1]))
            BtotalCh.append(BtotalTemp)
            if antibodies:
                AbTotalTemp = np.sum(challenge[jvals][:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
                challenge[jvals][0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(challenge[jvals][:, 1]))
                AbTotalCh.append(AbTotalTemp)
        challengeSolnsWithoutVaccine.append(challenge)
        BtotalSolnsWithoutVaccine.append(BtotalCh)
        AbTotalSolnsWithoutVaccine.append(AbTotalCh)


    # if antibodies:
    fig = plt.figure(myfig, figsize=[12, 10])
    # fig.text(0.37, 0.95, mymodelname, fontsize=12, fontweight='bold')
    fig.text(0.05, 0.92, 'A)', fontsize=12, fontweight="bold")
    fig.text(0.56, 0.92, 'B)', fontsize=12, fontweight="bold")
    fig.text(0.05, 0.45, 'C)', fontsize=12, fontweight="bold")
    fig.text(0.56, 0.45, 'D)', fontsize=12, fontweight="bold")
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.5, top=0.9, bottom=0.1)
    for mvals in range(2):
        for kvals in range(2):
            ax = plt.subplot(2, 2, kvals * 2 + mvals + 1)

            plt.plot(tspan0, (AbTotal[:] + 1), color=fourColorPalette[0], label='Primary infection', linewidth=3)
            plt.plot(tspan1, (vaccineAb[mvals][:] + 1), color=fourColorPalette[1], linewidth=3,  label='Vaccine') #+ r'$d=$' +
                            #str(int(relativeDistance(nIntervals,cloneVaclist[mvals],0))) + '%')

            plt.plot(tspan2, (AbTotalSolns[mvals][kvals][:] + 1), color=fourColorPalette[3], markevery=3, linewidth=3, label='Challenge ')# + r'$d=$' +
                            # str(int(relativeDistance(nIntervals,mydistances[kvals],0))) + '%')

            plt.plot(tspan2bis, (AbTotalSolnsWithoutVaccine[mvals][kvals][:] + 1), color=fourColorPalette[2], markevery=3, linewidth=3,
                     label='Challenge without vaccine') #+ r'$d=$' +

            plt.plot(tspan0, (Btotal[:] + 1), linestyle ="--", color=fourColorPalette[0], linewidth=3)
            plt.plot(tspan1, (vaccineB[mvals][:] + 1), '--', color=fourColorPalette[1], linewidth=3) #+ r'$d=$' +
                            #str(int(relativeDistance(nIntervals,cloneVaclist[mvals],0))) + '%')

            plt.plot(tspan2, (BtotalSolns[mvals][kvals][:] + 1), '--', color=fourColorPalette[3], markevery=3, linewidth=3)# + r'$d=$' +
                            # str(int(relativeDistance(nIntervals,mydistances[kvals],0))) + '%')

            plt.plot(tspan2bis, (BtotalSolnsWithoutVaccine[mvals][kvals][:] + 1), '--',color=fourColorPalette[2], markevery=3, linewidth=3,
                     ) #+ r'$d=$' +


                          # str(int(relativeDistance(nIntervals, mydistances[kvals], 0))) + '%')
            plt.axvline(x=30, color=fourColorPalette[1], linestyle=':', alpha=0.9, linewidth=2)

            plt.text(28, 10**6, 'Vaccination', rotation = 90)

            plt.axvline(x=60, color=fourColorPalette[3], linestyle=':', alpha=0.9, linewidth=2)

            plt.text(58, 10 ** 5, 'Challenge', rotation=90)

            plt.ylim([10 ** 0, 10 ** 9])
            plt.yscale('log')
            # plt.xticks(np.arange(0, max(tspan2) + 1, 10))
            loc1 = tspan0[4]
            loc2 = tspan1[0]
            loc3 = tspan2[0]
            plt.xticks([loc1, loc2, loc3])
            ax.set_xticklabels(['Primary \ninfection', 'Vaccination', 'Challenge'], fontsize=10, fontweight='bold')
            # plt.ylabel('Concentration', fontsize=12, fontweight='bold')
            # plt.legend(loc=1, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            # ax = plt.subplot(2, 2, kvals + 3)
            plt.xlabel('Days post-infection', fontsize=12, fontweight='bold')

            ax = plt.subplot(2, 2, 2 * kvals + 1)
            ax.set_ylabel("\n".join(wrap('d(Challenge, PI) = ' +
                                         str(int(relativeDistance(nIntervals, mydistances[kvals],
                                                                  0))) + '%\n Concentration', 30)),
                          fontweight='bold', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if mvals == 0:
                if kvals == 1:
                    plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=10, )

        ax = plt.subplot(2, 2, mvals + 1)
        ax.set_title("\n".join(wrap('d(Vaccine, PI) = ' +
                                    str(int(relativeDistance(nIntervals, cloneVaclist[mvals], 0))) + '% \n         ',
                                    30)),
                     fontweight='bold', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


    return fig




def figure4WindowsAll(myfunName, antibodies, initCond, Vindex, Bindex, params, params1, params2, nIntervals,
                                 hillParam1, hillParam2, cloneVaclist, cloneChallengelist, myfig, mymodelname):
    '''
    This figure will plot four windows to represent four scenarios:
    1) vaccine is close to Previous immunity AND challenge is close to previous immunity
    2) vaccine is close to Previous immunity AND challenge is Far from previous immunitya
    3) vaccine is FAR from previous immunity and challenge is CLOSE to previous immunity
    4) vaccine is FAR from previous immunity and challenge is far from previous immunity

    The results of this simulation also depend on the distance between the vaccine and the challenge.
    For each window, we will plot three curves:

     The first one is with the reference virus, the second one with a
     vaccine strain and the third one with a challenge.
    NOTE: in this function the B-cell equations and
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.



    :param myfunName: function describing the ODEs to use
    :param antibodies: receives 0 or 1, 0 if model doesn't include antibodies, 1 if it does
    :param initCond: Initial conditions for that particular set of ODEs
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the first B-cell clone
    :param params: parameters to be used for the wild-type virus
    :param params1: parameters to be used for an attenuated virus (the parameters themselves reflect the attenuation)
    :param params2: parameters to be used with a challenge virus
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param cloneVaclist: : a list with 2 numbers representing the distance between vaccine and reference virus
    :param cloneChallengelist: a list with 2 numbers representing the distance between challenge and reference virus
    :param myfig:  figure number
    :param mymodelname: the title of the figure
    :return: a figure handle

    NOTE: because of the way we are thinking distances, once we determine the distance from vaccine to reference and
    from challenge to reference, we have determined the distance from vaccine to challenge
    '''

    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    ##################### a typical influenza infection: #####################################
    # run this bit to get previous immunity
    ####Time parameters
    tstart0 = 0
    tfinal0 = 30
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    soln0 = odeint(myfunName, initCond, tspan0, args=(params, distanceMat[0, :], nIntervals))

    Btotal = np.sum(soln0[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln0[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln0[:, 1]))

    if antibodies:
        AbTotal = np.sum(soln0[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln0[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln0[:, 1]))

    ########################### Vaccination with LAIV #########################################
    # initial conditions:
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCond[ivals]

    tstart1 = 30
    tfinal1 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1 = odeint(myfunName, initCond1, tspan1,
                   args=(params1, distanceMat[cloneVaclist[0], :], nIntervals))

    Btotal1 = np.sum(soln1[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1[:, 1]))


    soln1bis = odeint(myfunName, initCond1, tspan1,
                   args=(params1, distanceMat[cloneVaclist[1], :], nIntervals))

    Btotal1bis = np.sum(soln1bis[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1bis[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1[:, 1]))

    if antibodies:
        AbTotal1 = np.sum(soln1[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
            soln1[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1[:, 1]))
        AbTotal1bis = np.sum(soln1bis[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln1bis[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1bis[:, 1]))

    vaccineSols = [soln1, soln1bis]
    vaccineAb = [AbTotal1, AbTotal1bis]
    vaccineB = [Btotal1, Btotal1bis]


    ################### Infection with a challenge after vaccination ##################################
    #Two strains given by the user:
    closestClone2 = cloneChallengelist[0]
    closestClone2_1 = cloneChallengelist[1]

    mydistances = [closestClone2, closestClone2_1]
    myadjectives = [' close', ' far']
    #run the challenge with each of the vaccine viruses:
    challengeSolns = []
    BtotalSolns = []
    AbTotalSolns = []
    for kvals in range(2):
        initCond2 = np.copy(vaccineSols[kvals][-1, :])
        for ivals in range(Bindex):
            initCond2[ivals] = initCond[ivals]

        tstart2 = 60
        tfinal2 = 90
        delta_t = 0.1
        num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
        tspan2 = np.linspace(tstart2, tfinal2, num_steps)

        soln2 = odeint(myfunName, initCond2, tspan2,
                       args=(params2, distanceMat[closestClone2, :], nIntervals))

        soln2_1 = odeint(myfunName, initCond2, tspan2,
                         args=(params2, distanceMat[closestClone2_1, :], nIntervals))


        challenge = [soln2, soln2_1]
        BtotalCh = []
        AbTotalCh = []

        for jvals in range(2):
            BtotalTemp = np.sum(challenge[jvals][:, Bindex:(Bindex + nIntervals)], 1) - np.sum(
                challenge[jvals][0, Bindex:(Bindex + nIntervals)]) * np.ones(
                np.size(challenge[jvals][:, 1]))
            BtotalCh.append(BtotalTemp)
            if antibodies:
                AbTotalTemp = np.sum(challenge[jvals][:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
                challenge[jvals][0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(challenge[jvals][:, 1]))
                AbTotalCh.append(AbTotalTemp)
        challengeSolns.append(challenge)
        BtotalSolns.append(BtotalCh)
        AbTotalSolns.append(AbTotalCh)


    #     ################### Infection with a challenge wihtout vaccination ##################################

    mydistances = [closestClone2, closestClone2_1]
    myadjectives = [' close', ' far']

    #run the challenge with each of the vaccine viruses:
    challengeSolnsWithoutVaccine = []
    BtotalSolnsWithoutVaccine = []
    AbTotalSolnsWithoutVaccine = []
    for kvals in range(2):
        initCond2bis = np.copy(soln0[-1, :])
        for ivals in range(Bindex):
            initCond2bis[ivals] = initCond[ivals]

        # print initCond2bis

        tstart2 = 60.9
        tfinal2 = 90.9
        delta_t = 0.1
        num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
        tspan2bis = np.linspace(tstart2, tfinal2, num_steps)

        soln2 = odeint(myfunName, initCond2bis, tspan2bis,
                       args=(params2, distanceMat[closestClone2, :], nIntervals))

        soln2_1 = odeint(myfunName, initCond2bis, tspan2bis,
                         args=(params2, distanceMat[closestClone2_1, :], nIntervals))


        challenge = [soln2, soln2_1]
        BtotalCh = []
        AbTotalCh = []

        for jvals in range(2):
            BtotalTemp = np.sum(challenge[jvals][:, Bindex:(Bindex + nIntervals)], 1) - np.sum(
                challenge[jvals][0, Bindex:(Bindex + nIntervals)]) * np.ones(
                np.size(challenge[jvals][:, 1]))
            BtotalCh.append(BtotalTemp)
            if antibodies:
                AbTotalTemp = np.sum(challenge[jvals][:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
                challenge[jvals][0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(challenge[jvals][:, 1]))
                AbTotalCh.append(AbTotalTemp)
        challengeSolnsWithoutVaccine.append(challenge)
        BtotalSolnsWithoutVaccine.append(BtotalCh)
        AbTotalSolnsWithoutVaccine.append(AbTotalCh)


    # if antibodies:
    fig = plt.figure(myfig, figsize=[12, 10])
    # fig.text(0.37, 0.95, mymodelname, fontsize=12, fontweight='bold')
    fig.text(0.05, 0.92, 'A)', fontsize=12, fontweight="bold")
    fig.text(0.56, 0.92, 'B)', fontsize=12, fontweight="bold")
    fig.text(0.05, 0.45, 'C)', fontsize=12, fontweight="bold")
    fig.text(0.56, 0.45, 'D)', fontsize=12, fontweight="bold")
    plt.subplots_adjust(left=0.1, right=0.95, wspace=0.3, hspace=0.5, top=0.9, bottom=0.1)
    for mvals in range(2):
        for kvals in range(2):

            ax = plt.subplot(2, 2, (2*kvals +1)  + mvals)
            plt.plot(tspan0, (soln0[:, Vindex] + 1), color=fourColorPalette[0], label='Primary infection', linewidth=3)
            plt.plot(tspan1, (vaccineSols[mvals][:, Vindex] + 1), color=fourColorPalette[1], linewidth=3,
                     label='Vaccine')  # + r'$d=$' +
            plt.plot(tspan2, (challengeSolns[mvals][kvals][:, Vindex] + 1), color=fourColorPalette[3], markevery=3,
                     linewidth=3, label='Challenge ')  # + r'$d=$' +
            plt.plot(tspan2bis, (challengeSolnsWithoutVaccine[mvals][kvals][:, Vindex] + 1), color=fourColorPalette[2],
                     markevery=3, linewidth=3,
                     label='Challenge without vaccine')  # + r'$d=$' +

            plt.axvline(x=30, color=fourColorPalette[1], linestyle=':', alpha=0.9, linewidth=2)

            # plt.text(28, 10**6, 'Vaccination', rotation = 90, fontsize=10)
            #
            plt.axvline(x=60, color=fourColorPalette[3], linestyle=':', alpha=0.9, linewidth=2)

            # plt.text(58, 10 ** 5, 'Challenge', rotation=90)

            plt.ylim([10 ** 0, 10 ** 9])
            plt.yscale('log')
            # plt.xticks(np.arange(0, max(tspan2) + 1, 10))
            loc1 = tspan0[4]
            loc2 = tspan1[0]
            loc3 = tspan2[0]
            plt.xticks([loc1, loc2, loc3])
            ax.set_xticklabels(['Primary \ninfection', 'Vaccination', 'Challenge'], fontsize=10, fontweight='bold')
            # plt.ylabel('Concentration', fontsize=12, fontweight='bold')
            # plt.legend(loc=1, fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')

            # ax = plt.subplot(2, 2, kvals + 3)
            plt.xlabel('Days post-infection', fontsize=12, fontweight='bold')

            ax = plt.subplot(2, 2, 2 * kvals + 1)
            ax.set_ylabel("\n".join(wrap('d(Challenge, PI) = ' +
                                         str(int(relativeDistance(nIntervals, mydistances[kvals],
                                                                  0))) + '%\n Concentration', 30)),
                          fontweight='bold', fontsize=12)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if mvals == 0:
                if kvals == 1:
                    plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=10, )

        ax = plt.subplot(2, 2, mvals + 1)
        ax.set_title("\n".join(wrap('d(Vaccine, PI) = ' +
                                    str(int(relativeDistance(nIntervals, cloneVaclist[mvals], 0))) + '% \n         ',
                                    30)),
                     fontweight='bold', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
    return fig




def computeFinalViralLoad(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1, hillParam2, myparams,
                          myparamsVac, closestClone1,
                          closestClone2):
    """
    This function will sequentially do the following:
    1) create preexisting immunity by simulating an infection with a "reference virus". We measure the distance from this
    virus to subsequent viruses/infections as the percentage difference in their composition (in some sort of metric):
    distance = 0 means that the two viruses perfectly match each other, 100% difference means that the immunity they ellicit
    is completely different and would not be protective against each other.

    2) simulate vaccination.
    3) simulate infection with a wild virus.

    computes the final viral load for a wild virus closest to closestClone2 and a vaccination virus closest to closestClone1

    NOTE: in this function the B-cell equations and 
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions 
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.


    :param myfunName: name of the ODE model
    :param initCond: initial conditions for the ode model
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the B-cells. 
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param myparams: parameters for the ODE for a wild-type virus
    :param myparamsVac: parameters for the ODE for an attenuated virus
    :param closestClone1: it is a number between 0 and nIntervals, it provides the distance between the reference virus and the virus
    contained in the vaccine
    :param closestClone2: it is a number between 0 and nIntervals, providing the distance between the reference virus and the wild
    virus used for the challenge.

    :return: the (logged) area under the curve for the infection with the challenge virus
    """

    # computes the distance matrix
    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)
    # print np.shape(distanceMat[0, :])

    # ####Parameters
    tstart0 = 0
    tfinal0 = 30
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:
    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:


    soln0 = odeint(myfunName, initCond, tspan0, args=(myparams, distanceMat[0, :], nIntervals))

    # vaccine virus
    # Reinitialize the number of uninfected cells, infected cells and virus (as present in the equations):
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCond[ivals]

    tstart1 = 30
    tfinal1 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1 = odeint(myfunName, initCond1, tspan1,
                   args=(myparamsVac, distanceMat[closestClone1, :], nIntervals))

    # finally, infection with a wild virus, that is somehow close to the vaccine:
    # Reinitialize the number of uninfected cells, infected cells and virus:
    initCond2 = soln1[-1, :]
    for ivals in range(Bindex):
        initCond2[ivals] = initCond[ivals]

    # print initCond2
    tstart2 = 60
    tfinal2 = 90
    delta_t = 0.1
    num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
    tspan2 = np.linspace(tstart2, tfinal2, num_steps)

    soln2 = odeint(myfunName, initCond2, tspan2,
                   args=(myparams, distanceMat[closestClone2, :], nIntervals))
    area2 = simps(soln2[:, Vindex], tspan2)
    return np.log10(area2 + 1)






def createVectorPercentages(myinputvector, nIntervals):
    totalIncrease = sum(myinputvector)

    percentage = np.zeros(nIntervals)
    for ivals in range(nIntervals):
        percentage[ivals] = (myinputvector[ivals] / totalIncrease)
    return percentage*totalIncrease








def figureUS_EuropeBarplots3WindowsAllClonesBis(myfunName, antibodies, initCond, Vindex, Bindex, params, params1, params2, nIntervals,
                                 hillParam1, hillParam2, clone2, clone3, myfig, mymodelname):
    """

   :param myfunName: name of the ODE model
    :param initCond: initial conditions for the ode model
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the B-cells.
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param myparams: parameters for the ODE for a wild-type virus
    :param myparamsVac: parameters for the ODE for an attenuated virus
    :param closestClone1: it is a number between 0 and nIntervals, it provides the distance between the reference virus and the virus
    contained in the vaccine
    :param closestClone2: it is a number between 0 and nIntervals, providing the distance between the reference virus and the wild
    virus used for the challenge.
    :return: a plot with 3 windows. The top left window will be a sequence of primary infection-vaccine-challenge like the
    plots for 3 consecutive infections, where the previous immunity and vaccine will be at distance closestclone1.
    The bottom left window will be a sequence of primary infection - vaccine - challenge, but I will put NO virus for the primary
    infection, so that the primary infection will effectively be the one with vaccine. this will simulate the situation
    in Europe, where children presumably had little pre-existing immunity. The graph on the right side plots the antibody
    total concentration after LAIV receipt for som
    """
    RGB_tuples = sns.color_palette("hls", nIntervals)
    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    ######################### simulate primary infection - vaccine - challenge #########################################
    ##################### a typical influenza infection: #####################################
    # run this bit to get previous immunity
    ####Time parameters
    tstart0 = 0
    tfinal0 = 100
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    soln0 = odeint(myfunName, initCond, tspan0, args=(params, distanceMat[0, :], nIntervals))

    Btotal = np.sum(soln0[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln0[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln0[:, 1]))

    if antibodies:
        AbTotal = np.sum(soln0[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln0[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln0[:, 1]))
    ########################### Vaccination with LAIV #########################################
    # initial conditions:
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCond[ivals]

    tstart1 = 30
    tfinal1 = 130
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1_us = odeint(myfunName, initCond1, tspan1,
                   args=(params1, distanceMat[clone2, :], nIntervals))

    Btotal1_us = np.sum(soln1_us[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1_us[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1_us[:, 1]))

    relativeIncreaseB1_us = soln1_us[-1, Bindex:Bindex + nIntervals] - \
                         soln1_us[0, Bindex:Bindex + nIntervals]

    perB_us1 = createVectorPercentages(relativeIncreaseB1_us, nIntervals)
    if antibodies:
        AbTotal1_us = np.sum(soln1_us[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln1_us[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1_us[:, 1]))

        relativeIncreaseAntibodies1_us = soln1_us[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                      soln1_us[0, Bindex + nIntervals:Bindex + 2 * nIntervals]

        perAb_us = createVectorPercentages(relativeIncreaseAntibodies1_us, nIntervals)
        # print percentageAb_us*100


    ################### Infection with a challenge after vaccination ##################################
    #Three strains given by the user:
    closestClone2 = clone3

    initCond2 = np.copy(soln1_us[-1, :])
    for ivals in range(Bindex):
        initCond2[ivals] = initCond[ivals]

    tstart2 = 60
    tfinal2 = 160
    delta_t = 0.1
    num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
    tspan2 = np.linspace(tstart2, tfinal2, num_steps)

    soln2_us = odeint(myfunName, initCond2, tspan2,
                   args=(params2, distanceMat[closestClone2, :], nIntervals))

    Btotal2_us = np.sum(soln2_us[:, Bindex:(Bindex + nIntervals)], 1) - \
                    np.sum(soln2_us[0, Bindex:(Bindex + nIntervals)]) * np.ones(np.size(soln2_us[:, 1]))

    relativeIncreaseB2_us = soln2_us[-1, Bindex:Bindex + nIntervals] - \
                            soln2_us[0, Bindex:Bindex + nIntervals]

    perB_us2 = createVectorPercentages(relativeIncreaseB2_us, nIntervals)


    if antibodies:
        AbTotal2_us = np.sum(soln2_us[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
            soln2_us[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln2_us[:, 1]))

        relativeIncreaseAntibodies2_us = soln2_us[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                     soln2_us[0, Bindex + nIntervals:Bindex + 2 * nIntervals]

        perAb_us2 = createVectorPercentages(relativeIncreaseAntibodies2_us, nIntervals)

    ######################### simulate NO primary infection - vaccine - challenge #####################################

    ########################### Vaccination with LAIV using the initial conditions given by user #########################################
    # initial conditions:
    initCond1 = np.copy(initCond)

    tstart1 = 30
    tfinal1 = 130
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1_eu = odeint(myfunName, initCond1, tspan1, args=(params1, distanceMat[clone2, :], nIntervals))

    Btotal1_eu = np.sum(soln1_eu[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln1_eu[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln1_eu[:, 1]))

    relativeIncreaseB1_eu = soln1_eu[-1, Bindex:Bindex + nIntervals] - \
                           soln1_eu[0, Bindex:Bindex + nIntervals]

    perB_eu1 = createVectorPercentages(relativeIncreaseB1_eu, nIntervals)

    if antibodies:
        AbTotal1_eu = np.sum(soln1_eu[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln1_eu[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln1_eu[:, 1]))

        relativeIncreaseAntibodies1_eu = soln1_eu[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                         soln1_eu[0, Bindex + nIntervals:Bindex + 2 * nIntervals]
        perAb_eu = createVectorPercentages(relativeIncreaseAntibodies1_eu, nIntervals)
    ################### Infection with a challenge after vaccination ##################################
    #Three strains given by the user:
    closestClone2 = clone3

    initCond2 = np.copy(soln1_eu[-1, :])
    for ivals in range(Bindex):
        initCond2[ivals] = initCond[ivals]

    tstart2 = 60
    tfinal2 = 160
    delta_t = 0.1
    num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
    tspan2 = np.linspace(tstart2, tfinal2, num_steps)

    soln2_eu = odeint(myfunName, initCond2, tspan2,
                   args=(params2, distanceMat[closestClone2, :], nIntervals))

    Btotal2_eu = np.sum(soln2_eu[:, Bindex:(Bindex + nIntervals)], 1) - np.sum(soln2_eu[0, Bindex:(Bindex + nIntervals)]) * np.ones(
        np.size(soln2_eu[:, 1]))

    relativeIncreaseB2_eu = soln2_eu[-1, Bindex:Bindex + nIntervals] - \
                            soln2_eu[0, Bindex:Bindex + nIntervals]

    if antibodies:
        AbTotal2_eu = np.sum(soln2_eu[:, (Bindex + nIntervals):(Bindex + 2 * nIntervals)], 1) - np.sum(
        soln2_eu[0, (Bindex + nIntervals):(Bindex + 2 * nIntervals)]) * np.ones(np.size(soln2_eu[:, 1]))

        relativeIncreaseAntibodies2_eu = soln2_eu[-1, Bindex + nIntervals:Bindex + 2 * nIntervals] - \
                                     soln2_eu[0, Bindex + nIntervals:Bindex + 2 * nIntervals]

        perAb_eu2 = createVectorPercentages(relativeIncreaseAntibodies2_eu, nIntervals)

    #compute the magnitude of the viral load generated by LAIV:
    LAIV_us = simps(soln1_us[:, Vindex], tspan1)
    LAIV_eu = simps(soln1_eu[:, Vindex], tspan1)

    #compute the magnitude of the response generated by LAIV, as the area under the curve for the total response in
    #B-cells and antibodies:
    BLAIV_us = simps(Btotal1_us, tspan1)
    BLAIV_eu = simps(Btotal1_eu, tspan1)


    AbLAIV_us = simps(AbTotal1_us, tspan1)
    AbLAIV_eu = simps(AbTotal1_eu, tspan1)


    ############################# PLOTTING: ##################################################################
    fig = plt.figure(myfig, figsize=(10, 8))
    fig.text(0.06, 0.93, 'A)', fontsize=12, fontweight="bold")
    fig.text(0.06, 0.47, 'B)', fontsize=12, fontweight="bold")
    fig.text(0.48, 0.93, 'C)', fontsize=12, fontweight="bold")
    # fig.text(0.66, 0.93, 'D)', fontsize=12, fontweight="bold")

    # fig.text(0.66, 0.93, 'D)', fontsize=12, fontweight="bold")
    plt.subplots_adjust(left=0.1, right=0.85, wspace=0.48, hspace=0.4, top=0.9, bottom=0.1)
    # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6),sharey=True)
    # N = 1
    # ind = np.array(['US', 'UK'])  # the x locations for the groups
    # width = 0.35
    # y = [BLAIV_us, BLAIV_eu]
    # y2= [AbLAIV_us, AbLAIV_eu]
    # r1 = np.arange(1, nIntervals + 1)


    loc1 = 1
    loc2 = 1.75
    myticks = [1, 1.75]
    mywidth = 0.45

    ax = plt.subplot2grid((2,2), (0,0))
    if antibodies:
        plt.plot(tspan0[:150], (soln0[:150, Vindex] + 1), color=twoColorPalette1[0], label='Primary infection', linewidth=3)
        plt.plot(tspan1[:150], (soln1_us[:150, Vindex] + 1),'--', color=twoColorPalette1[0], label='Vaccine', linewidth=3)
        plt.plot(tspan2[:150], (soln2_us[:150, Vindex] + 1),'o', color=twoColorPalette1[0], markevery=3, linewidth=3, label='Challenge')
        plt.axvline(x=30, color=fourColorPalette[1], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(25, 10 ** 2, 'Vaccination', rotation=90, fontsize=10)
        plt.axvline(x=60, color=fourColorPalette[1], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(55, 10 ** 2, 'Challenge', rotation=90, fontsize=10)
        plt.axvline(x=0, color=fourColorPalette[1], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(-5, 2*10 ** 3, 'Primary infection', rotation=90, fontsize=10)
        plt.ylim([10 ** 0, 10 ** 7])
        plt.xlim([-10, 80])
        plt.yscale('log')
        plt.xticks([])
        # plt.xticks([0, 30, 60], ['Primary \ninfection','Vaccination', 'Challenge'], rotation=45)
        plt.ylabel('Viral Concentration', fontsize=12, fontweight='bold')
        plt.xlabel('Time (days post infection)', fontsize=12, fontweight='bold')
        plt.legend(loc=1, fontsize=12)
        plt.title('US', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    ax = plt.subplot2grid((2, 2), (1, 0))
    if antibodies:
        # plt.plot(tspan0, (soln0[:, Vindex] + 1), color=twoColorPalette1[1], label='Primary infection', linewidth=2)
        plt.plot(tspan1[:150], (soln1_eu[:150, Vindex] + 1), '--',  color=twoColorPalette1[1], label='Vaccine', linewidth=3)
        plt.plot(tspan2[:150], (soln2_eu[:150, Vindex] + 1), 'o',  color=twoColorPalette1[1], markevery=3, linewidth=3, label='Challenge')

        plt.axvline(x=30, color=fourColorPalette[3], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(25, 10 ** 2, 'Vaccination', rotation=90, fontsize=10, fontweight='bold' )
        plt.axvline(x=60, color=fourColorPalette[3], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(55, 10 ** 2, 'Challenge', rotation=90, fontsize=10, fontweight='bold')
        plt.axvline(x=0, color=fourColorPalette[3], linestyle=':', alpha=0.9, linewidth=2)
        plt.text(-5, 2*10 ** 3, 'Primary infection', rotation=90, fontsize=10, fontweight='bold')
        plt.ylim([10 ** 0, 10 ** 7])
        plt.xlim([-10, 80])
        plt.xticks([])
        plt.yscale('log')
        plt.ylabel('Viral Concentration', fontsize=12, fontweight='bold')
        plt.xlabel('Time (days post infection)', fontsize=12, fontweight='bold')
        plt.legend(loc=1, fontsize=12)
        plt.title('UK',fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')


    ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    plt.bar(loc1, perAb_us[0], color=RGB_tuples[0], width=mywidth)
    mysum = perAb_us[0]
    for ivals in range(1, nIntervals):
        plt.bar(loc1, perAb_us[ivals], bottom = mysum, color=RGB_tuples[ivals], width=mywidth)
        mysum = mysum + perAb_us[ivals]


    plt.bar(loc2, perAb_eu[0], color=RGB_tuples[0], width=mywidth, label='Clone ' + str(1))
    mysum = perAb_eu[0]
    for ivals in range(1, nIntervals):
        plt.bar(loc2, perAb_eu[ivals], bottom = mysum, color=RGB_tuples[ivals], width=mywidth, label='Clone ' + str(ivals+1))
        mysum = mysum + perAb_eu[ivals]
    # plt.ylim([0,100])
    plt.yscale('log')
    plt.ylim([10 ** 0, 10 ** 7])
    plt.xticks(myticks)
    ax.set_xticklabels(['US', 'UK'], fontsize=12, fontweight='bold')
    # plt.title('Magnitude of antibody response \n after LAIV receipt')
    plt.ylabel('Antibody Total Concentration after LAIV receipt', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(bbox_to_anchor=(1.2, 1.05), loc=9,
              ncol=1, fontsize=12)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    return fig




def computeVE(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1, hillParam2, myparams,
                          myparamsVac, closestClone1,
                          closestClone2):
    """
    This function will compute a within-host VE by doing the following:
    1) create preexisting immunity by simulating an infection with a "reference virus". We measure the distance from this
    virus to subsequent viruses/infections as the percentage difference in their composition (in some sort of metric):
    distance = 0 means that the two viruses perfectly match each other, 100% difference means that the immunity they ellicit
    is completely different and would not be protective against each other.

    2) simulate vaccination.
    3) simulate infection with a wild virus.
    4) simulate an infection without vaccination
    5) compute the AUC for each of 3 and 4
    6) compute the ratio 1 - step3/step4

    computes the final viral load for a wild virus closest to closestClone2 and a vaccination virus closest to closestClone1

    NOTE: in this function the B-cell equations and
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.


    :param myfunName: name of the ODE model
    :param initCond: initial conditions for the ode model
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the B-cells.
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param myparams: parameters for the ODE for a wild-type virus
    :param myparamsVac: parameters for the ODE for an attenuated virus
    :param closestClone1: it is a number between 0 and nIntervals, it provides the distance between the reference virus and the virus
    contained in the vaccine
    :param closestClone2: it is a number between 0 and nIntervals, providing the distance between the reference virus and the wild
    virus used for the challenge.

    :return: the (logged) area under the curve for the infection with the challenge virus
    """

    # computes the distance matrix
    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    # ####Parameters
    tstart0 = 0
    tfinal0 = 30
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:
    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:


    soln0 = odeint(myfunName, initCond, tspan0, args=(myparams, distanceMat[0, :], nIntervals))

    # vaccine virus
    # Reinitialize the number of uninfected cells, infected cells and virus (as present in the equations):
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCond[ivals]

    tstart1 = 30
    tfinal1 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1 = odeint(myfunName, initCond1, tspan1,
                   args=(myparamsVac, distanceMat[closestClone1, :], nIntervals))

    # finally, infection with a wild virus, that is somehow close to the vaccine:
    # Reinitialize the number of uninfected cells, infected cells and virus:
    initCond2 = soln1[-1, :]
    for ivals in range(Bindex):
        initCond2[ivals] = initCond[ivals]

    # print initCond2
    tstart2 = 60
    tfinal2 = 90
    delta_t = 0.1
    num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
    tspan2 = np.linspace(tstart2, tfinal2, num_steps)

    soln2 = odeint(myfunName, initCond2, tspan2,
                   args=(myparams, distanceMat[closestClone2, :], nIntervals))
    areaV = simps(soln2[:, Vindex], tspan2)

    #run an infection without vaccination

    initCond3 = soln0[-1, :]
    for ivals in range(Bindex):
        initCond3[ivals] = initCond[ivals]

    # print initCond2
    tstart3 = 60
    tfinal3 = 90
    delta_t = 0.1
    num_steps = np.floor((tfinal3 - tstart3) / delta_t) + 1
    tspan3 = np.linspace(tstart3, tfinal3, num_steps)

    soln3 = odeint(myfunName, initCond3, tspan3,
                   args=(myparams, distanceMat[closestClone2, :], nIntervals))
    area3 = simps(soln3[:, Vindex], tspan3)

    if area3 != 0:
        return (1.0 - (areaV/area3))
    else:
        return -1


def computeVE2(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1, hillParam2, myparams,
                          myparamsVac, closestClone1,
                          closestClone2, initCondbis):
    """
    this function is different from computeVE in that it allows for no pre-existing immunity, by adding a second set of
    initial conditions, initCondbis.
    So that initCond will be the initial conditions for pre-existing immunity (so if we want not to have any pre-existing
    immunity we just have to set V=0 in this set) and initCondbis will be the initial conditions used for the other
    infections (vaccine and challenge). so that here, we can set V=1
    This function will compute a within-host VE by doing the following:
    1) create preexisting immunity by simulating an infection with a "reference virus". We measure the distance from this
    virus to subsequent viruses/infections as the percentage difference in their composition (in some sort of metric):
    distance = 0 means that the two viruses perfectly match each other, 100% difference means that the immunity they ellicit
    is completely different and would not be protective against each other.

    2) simulate vaccination.
    3) simulate infection with a wild virus.
    4) simulate an infection without vaccination
    5) compute the AUC for each of 3 and 4
    6) compute the ratio 1 - step3/step4

    computes the final viral load for a wild virus closest to closestClone2 and a vaccination virus closest to closestClone1

    NOTE: in this function the B-cell equations and
    antibody equations need to be the last equations in the model. This is because we will reset the initial conditions
    for ALL the equations EXCEPT for the B-cells and antibodies to simulate new infections with an attenuated virus or a
    challenge.


    :param myfunName: name of the ODE model
    :param initCond: initial conditions for the ode model
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the B-cells.
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param myparams: parameters for the ODE for a wild-type virus
    :param myparamsVac: parameters for the ODE for an attenuated virus
    :param closestClone1: it is a number between 0 and nIntervals, it provides the distance between the reference virus and the virus
    contained in the vaccine
    :param closestClone2: it is a number between 0 and nIntervals, providing the distance between the reference virus and the wild
    virus used for the challenge.

    :return: the (logged) area under the curve for the infection with the challenge virus
    """

    # computes the distance matrix
    distanceMat = createDistanceMatrix(nIntervals, hillParam1, hillParam2)

    # ####Parameters
    tstart0 = 0
    tfinal0 = 30
    delta_t = 0.1
    num_steps = np.floor((tfinal0 - tstart0) / delta_t) + 1
    tspan0 = np.linspace(tstart0, tfinal0, num_steps)

    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:
    # sequentially infecting someone with different viruses:
    # first virus is closest to clone 0 and we start with 1 B-cell in each clone:


    soln0 = odeint(myfunName, initCond, tspan0, args=(myparams, distanceMat[0, :], nIntervals))

    # vaccine virus
    # Reinitialize the number of uninfected cells, infected cells and virus (as present in the equations):
    initCond1 = np.copy(soln0[-1, :])
    for ivals in range(Bindex):
        initCond1[ivals] = initCondbis[ivals]      #use the other set of initial conditions

    tstart1 = 30
    tfinal1 = 60
    delta_t = 0.1
    num_steps = np.floor((tfinal1 - tstart1) / delta_t) + 1
    tspan1 = np.linspace(tstart1, tfinal1, num_steps)

    soln1 = odeint(myfunName, initCond1, tspan1,
                   args=(myparamsVac, distanceMat[closestClone1, :], nIntervals))

    # finally, infection with a wild virus, that is somehow close to the vaccine:
    # Reinitialize the number of uninfected cells, infected cells and virus:
    initCond2 = soln1[-1, :]
    for ivals in range(Bindex):
        initCond2[ivals] = initCondbis[ivals]     #use the other set of initial conditions

    # print initCond2
    tstart2 = 60
    tfinal2 = 90
    delta_t = 0.1
    num_steps = np.floor((tfinal2 - tstart2) / delta_t) + 1
    tspan2 = np.linspace(tstart2, tfinal2, num_steps)

    soln2 = odeint(myfunName, initCond2, tspan2,
                   args=(myparams, distanceMat[closestClone2, :], nIntervals))
    areaV = simps(soln2[:, Vindex], tspan2)

    #run an infection without vaccination

    initCond3 = soln0[-1, :]
    for ivals in range(Bindex):
        initCond3[ivals] = initCondbis[ivals]      #use the other set of initial conditions

    # print initCond2
    tstart3 = 60
    tfinal3 = 90
    delta_t = 0.1
    num_steps = np.floor((tfinal3 - tstart3) / delta_t) + 1
    tspan3 = np.linspace(tstart3, tfinal3, num_steps)

    soln3 = odeint(myfunName, initCond3, tspan3,
                   args=(myparams, distanceMat[closestClone2, :], nIntervals))
    area3 = simps(soln3[:, Vindex], tspan3)

    if area3 != 0:
        return (1.0 - (areaV/area3))
    else:
        return -1




def contourPlotsAll(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1, hillParam2, myparams,
                               myparamsVac,
                               modelname, fig):
    """
    :param myfunName: name of the ODE model
    :param initCond: initial conditions for the ode model
    :param Vindex: index of the variable representing Virus
    :param Bindex: index of the variable representing the B-cells.
    :param nIntervals: number of clones
    :param hillParam1: parameter for Hill equation
    :param hillParam2: parameter for Hill equation
    :param myparams: parameters for the ODE for a wild-type virus
    :param myparamsVac: parameters for the ODE for an attenuated virus
    :param modelname: Name of the model that will be used.
    :param fig: figure number
    :return: Returns a figure and figure handle with two windows:
    Left window plots the contour plot of the severity of infection with a challenge strain as a function of two variables:
    -the antigenic distance from challenge to pre-existing immunity and
    -the antigenic distance from vaccine to pre-existing immunity
    the severity of infection is calculated as the (log) area under the curve (AUC) of the challenge infection.

    Right window: plots the within-host vaccine effect, defined as 1 - AUC_with vaccination/AUC_without vaccination,
    again, as a function of the two antigenic distances defined for the left panel.
    """
    contourPalette = sns.color_palette(sns.color_palette("coolwarm", nIntervals+5))
    contourPalette1 = sns.diverging_palette(160, 290, s=100, l=45,n=nIntervals)#sns.color_palette("BrBG", nIntervals+5)

    numClones = 100 / nIntervals
    fig1 = plt.figure(fig, figsize=(10, 10))
    fig1.text(0.03, 0.72, 'A) Extent of viral replication',  fontsize=12, fontweight="bold")
    fig1.text(0.06, 0.70, 'of challenge infection',  fontsize=12, fontweight="bold")
    fig1.text(0.53, 0.72, 'B) Within-host vaccine effect ' + r'$\boldsymbol{ve}_{\omega}$', fontsize=12, \
              fontweight="bold")


    plt.subplots_adjust(wspace=0.5, hspace=0.3, left=0.1, right=0.95, bottom=0.05, top=0.95)


    clone1 = np.linspace(0,100, nIntervals)
    clone2 = np.linspace(0, 100, nIntervals)
    clone1, clone2 = np.meshgrid(clone1, clone2, indexing='ij')

    diag = np.linspace(0, 100, 6)

    # Define a matrix where we will store the results for each pair ij:
    mysol1 = np.zeros((nIntervals, nIntervals))
    mysol2 = np.zeros((nIntervals, nIntervals))

    for xvals in range(nIntervals):
        for yvals in range(nIntervals):
            # print [ivals, jvals]
            mysol1[xvals, yvals] = \
            computeFinalViralLoad(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1,
                                                        hillParam2, myparams, myparamsVac, xvals,
                                                        yvals)


            mysol2[xvals, yvals] = \
            computeVE(myfunName, initCond, Vindex, Bindex, nIntervals, hillParam1,
                                            hillParam2, myparams, myparamsVac, xvals, yvals)

    ax = plt.subplot(1, 2, 1, aspect=1)
    cs = plt.contourf(clone1, clone2, mysol1, 25, cmap=ListedColormap(contourPalette))  # as_cmap=True))
    plt.plot(diag, diag, 'k', linewidth=2)
    plt.xlabel('Vaccine distance to Pre-existing immunity \n' + r'$d(V, PI)$', fontsize=12, fontweight="bold")
    plt.ylabel('Challenge distance to Pre-existing immunity\n'+ r'$d(C, PI)$', fontsize=12, fontweight="bold")
    plt.xticks(np.linspace(0, 100, 6))
    plt.yticks(np.linspace(0, 100, 6))
    plt.text(50,55, 'Perfectly matched vaccine', horizontalalignment='center',
    verticalalignment='center', rotation=45,fontsize=10, fontweight='bold')


    cbar = plt.colorbar(cs, orientation='horizontal') #fraction=0.046, pad=0.04,
    cbar.ax.set_xlabel('Log' + r'$_{10}$' + ' Viral load challenge virus', fontsize=10, fontweight="bold")

    plt.subplot(1, 2, 2, aspect=1)
    cs1 = plt.contourf(clone1, clone2, mysol2, 25, cmap=ListedColormap(contourPalette1))# as_cmap=True))
    plt.plot(diag, diag, 'k', linewidth=2)
    plt.xlabel('Vaccine distance to Pre-existing immunity \n' + r'$d(V, PI)$', fontsize=12, fontweight="bold")
    plt.ylabel('Challenge distance to Pre-existing immunity\n' + r'$d(C, PI)$', fontsize=12, fontweight="bold")
    plt.xticks(np.linspace(0,100, 6))
    plt.yticks(np.linspace(0, 100, 6))
    plt.text(50, 55, 'Perfectly matched vaccine', horizontalalignment='center',
             verticalalignment='center', rotation=45, fontsize=10, fontweight='bold')
    cbar1 = plt.colorbar(cs1, orientation='horizontal')#fraction=0.046, pad=0.04,
    cbar1.ax.set_xlabel('Within-host vaccine effect ' + r'$\boldsymbol{ve}_{\omega}$', fontsize=10, fontweight="bold")

    return fig1



