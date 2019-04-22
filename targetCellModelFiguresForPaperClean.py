from __future__ import division
import sys
import numpy as np
import csv
import colorsys
import matplotlib
import time
from scipy import sparse
from scipy.integrate import odeint, ode, simps, trapz
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


# from brokenaxes import brokenaxes



import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from hillFunction import hillFunction, hillFunctionCentered
from createMatrices import createDistanceMatrix, relativeDistance
from withinHostEquations import influenzaWithinHostEquations5bis, influenzaWithinHostEquations8, \
    influenzaWithinHostEquations9, influenzaWithinHostEquations10, influenzaWithinHostEquations12
from textwrap import wrap
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

from genericFunctionsForDifferentEquationsClean import *
from withinHostEquations import influenzaWithinHostEquations8, influenzaWithinHostEquations8_normalizeNumClones


today = time.strftime("%d%b%Y", time.localtime())
#parameters:
#General parameters:
nIntervals = 20
hillParam1 = 10
hillParam2 = 15

numClones = (100 / nIntervals)
redBeta = 1
redk = 25

#Parameters (original)
beta = 2.7 * 10 ** (-5)
d_A = 10 ** (-1)
d_I = 4.0 * 10 ** (0)
d_V = 3.0
k = 5 * 10 ** (-3)
p_B = 1 * 10 ** (-1)
p_I = 1.2 * 10 ** (-2)
phi = 5 * 10 ** 3
sigma = 10



params8 = [beta, d_A, d_I, d_V, k, p_B, p_I, phi, sigma]



betaVac = beta*redBeta
kVac = k*redk
paramsVac8 = [betaVac, d_A, d_I, d_V, kVac, p_B, p_I, phi, sigma]


U0 = 4 * 10 ** 8
I0 = 0
V0 = 9.3 * 10 ** (-2)
initCond8 = [U0, I0, V0]
for i in range(nIntervals):
    initCond8.append(1)
for i in range(nIntervals):
    initCond8.append(0)

initCondWithoutPI = np.copy(initCond8)
initCondWithoutPI[2] = 0

cloneVaclist = [1, 6]; cloneChallengelist= [2, 7]

mylabels = ["U", "I", "V"]
# #### FIGURE 1 #####
# myfig1 = viralDynamicsPrimaryInfection2PieCharts(influenzaWithinHostEquations8, 1, initCond8, 3, mylabels, params8,
#                                                     paramsVac8,
#                                                     nIntervals, hillParam1, hillParam2, 1, 'Target-cell model')

# myfilename = 'figures/paperTargetCellFigures/viralDynamicsPrimaryInfection2PieCharts' + '_kVacMultiplied_by_' + str(redk) +\
#      '_betaVacMultiplied_by_' + str(redBeta) + today +  '.eps'
#
# myfig1.savefig(myfilename)

#### FIGURE 2 #####
# myfig2 = figurePreviousImmunityVaccineGenericStackedBars(influenzaWithinHostEquations8, 1, initCond8, 2, 3,
#                                      params8, paramsVac8,nIntervals,
#                                          hillParam1, hillParam2, [2, 9,10], 2)
#
# myfilename = 'figures/paperTargetCellFigures/figurePreviousImmunityVaccineGenericStackedBars' + '_kVacMultiplied_by_' + str(redk) +\
#      '_betaVacMultiplied_by_' + str(redBeta) + today +  '.eps'
#
# myfig2.savefig(myfilename)

### FIGURE 3 #####
# myfig3 = contourPlotsAll(influenzaWithinHostEquations8, initCond8, 2, 3,
#                                   nIntervals, hillParam1, hillParam2, params8, paramsVac8,
#                                'Target-cell model', 3)
#
# myfilename = 'figures/paperTargetCellFigures/contourPlotsBoth' + '_kVacMultiplied_by_' + str(redk) +\
#      '_betaVacMultiplied_by_' + str(redBeta) + today +  '.eps'
#
# myfig3.savefig(myfilename)



# #### FIGURE 5 #####
# myfig5 =  figureUS_EuropeBarplots3WindowsAllClonesBis(influenzaWithinHostEquations8, 1, initCond8, 2, 3,
#                                      params8, paramsVac8, params8, nIntervals,
#                                          hillParam1, hillParam2,  0, 4, 4, 'Target-cell model')
#
#
#
# myfilename = 'figures/paperTargetCellFigures/figureUS_EuropeBarplot3WindowsAllClonesBis' + '_kVacMultiplied_by_' + str(redk) +\
#      '_betaVacMultiplied_by_' + str(redBeta) + today + '.eps'
# print myfilename
# myfig5.savefig(myfilename)




# ##### SUPPLEMENTAL FIGURES ####
#
myfig5 = figure4WindowsAll(influenzaWithinHostEquations8_normalizeNumClones, 1, initCond8, 2, 3,
                              params8, paramsVac8, params8, nIntervals, hillParam1, hillParam2,
                              cloneVaclist, cloneChallengelist, 5, 'Target-cell limitation model')
# # # myfilename = 'figures/paperTargetCellFigures/figure4windowsAll_VandC_close_' + '_kVacMultiplied_by_' + str(redk) +\
# # #      '_betaVacMultiplied_by_' + str(redBeta) + 'Revised.pdf'
# # # myfig5.savefig(myfilename)
# #
# myfig6 = figure4WindowsAntibodies(influenzaWithinHostEquations8, 1, initCond8, 2, 3,
#                               params8, paramsVac8, params8, nIntervals, hillParam1, hillParam2,
#                               cloneVaclist, cloneChallengelist, 6, 'Target-cell limitation model')
#





#compute the ve_w for the US case and for the EU case:
#US case: NO pre-existing conditions
# eu_ve = computeVE2(influenzaWithinHostEquations8, initCondWithoutPI, 2, 3,
#                                    nIntervals, hillParam1, hillParam2, params8, paramsVac8, 0,
#                           4, initCond8)
# print eu_ve
#
# us_ve = computeVE2(influenzaWithinHostEquations8, initCond8, 2, 3,
#                                    nIntervals, hillParam1, hillParam2, params8, paramsVac8, 0,
#                           4, initCond8)
# print us_ve




plt.show()