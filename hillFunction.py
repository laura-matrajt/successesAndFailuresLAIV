'''
Created on Sept 20th, 2016
this file will create a Hill function that will relate antigenic distance to antibody performance
@author: Laura
'''
from __future__ import division
import sys
import time
import math
import pickle
import numpy as np
import colorsys
from scipy import sparse
from scipy.integrate import odeint, simps
from matplotlib import pyplot as plt

def hillFunction(x, K, n, Vmax=-1):
    """
    Hill function is given by: f(x) = Vmax*s^n/(K^n + s^n)
    :param x: percentage of difference between two antigens/antigen-antibody in some distance
    :param k:
    :param n:
    :param Vmax:
    :return: the value of the Hill function at each point x
    """
    val = (Vmax*x**n)/(K**n + x**n) + 1
    return val



if __name__ == "__main__":
    Ncol = 30
    HSV_tuples = [(x * 1.0 / Ncol, 0.5, 0.5) for x in range(Ncol)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    Vmax = -1
    mycolors = ['b', 'g']
    x = np.linspace(0,100,100)
    K = 10
    n = (5)
    for ivals in range(2):
        myk = [10, 27]
        K = myk[ivals]
        y = hillFunction(x, K, n)
        #plt.subplot(1,2,ivals+1)
        #plt.plot(x,y, color = RGB_tuples[ivals], label = "K="+ str(K), linewidth=3)
        plt.plot(x, y, color=mycolors[ivals], label = "K="+ str(K), linewidth=3)
    plt.xlabel("Antigenic distance (%)", fontweight='bold',fontsize=14)
    plt.ylabel("Antibody binding efficacy", fontweight='bold',fontsize=14)
    plt.legend()
    plt.savefig("functionAntigenicDistance.pdf")
    plt.show()