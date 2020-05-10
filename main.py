import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as anim
from matplotlib.collections import LineCollection
import scipy.integrate as ode

import math

from vivelesmaths import *

from multiprocessing import Process

import csv
import os

import time

plt.style.use('dark_background')
# matplotlib.rcParams['text.usetex'] = True

# données-------------------------------------

m_0 = 3.003451249261589
p_0 = 3.645639392529623
k = 0.026542410398766054
m_1 = 0.6946390572490012
l = 0.2617581820549962
p_1 = 0.1274926083122092

# --------------------------------------------

Pr = 1

def dF(_, F):
    return np.array([
        F[1],
        F[2],
        2 * F[1] ** 2 - 3 * F[0] * F[2] - F[3],
        F[4],
        - 3 * Pr * F[0] * F[4]
    ])


def rk4(dF, x_0, a, h):
    imax = int((x_0[1] - x_0[0]) / h)
    X = np.arange(imax + 1) * h
    U = np.zeros((imax + 1, 5))
    U[0, :] = [0, 0, a[0], 1, a[1]]
    for i in range(imax):
        K1 = dF(i * h, U[i, :])
        K2 = dF((i + 0.5) * h, U[i, :] + K1 * h / 2)
        K3 = dF((i + 0.5) * h, U[i, :] + K2 * h / 2)
        K4 = dF((i + 1) * h, U[i, :] + K3 * h)
        U[i + 1, :] = U[i, :] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6
    return X, U


def find_da(J, alpha):
    return Gauss(-J, alpha)


# ode.solve_ivp(dF, (0, 5), Y, max_step=1/500)
"""
def step(a):
    da = 0.1

    F = np.array([0, 0, a[0], 1, a[1]])
    F_a0 = np.array([0, 0, a[0] + da, 1, a[1]])
    F_a1 = np.array([0, 0, a[0], 1, a[1] + da])

    Y = ode.solve_ivp(dF, (0, 5), F, max_step=1 / 500)
    Y_a0 = ode.solve_ivp(dF, (0, 5), F_a0, max_step=1 / 500)
    Y_a1 = ode.solve_ivp(dF, (0, 5), F_a1, max_step=1 / 500)

    alpha = np.array([Y.y[1][-1], Y.y[3][-1]])
    alpha_a0 = [Y_a0.y[1][-1], Y_a0.y[3][-1]]
    alpha_a1 = [Y_a1.y[1][-1], Y_a1.y[3][-1]]

    J = np.array([
        [(alpha_a0[0] - alpha[0]) / da, (alpha_a1[0] - alpha[0]) / da],
        [(alpha_a0[1] - alpha[1]) / da, (alpha_a1[1] - alpha[1]) / da]
    ])
    print(J)
    print(alpha)
    da = find_da(J, alpha)
    print("a = ", a + da)
    return a + da

error = 1

def solver(Pr=1, error=1, N_max=20):
    a = [-2, 0.1]
    a_new = step(a)
    it = 1
    while (abs(a[0] - a_new[0]) > error or abs(a[1] - a_new[1]) > error) and it <= N_max:
        a = np.copy(a_new)
        a_new = step(a)
        it += 1
    if it <= N_max:
        print(a_new)
    else:
        print("bah c'est baisé...")
    F = np.array([0, 0, a_new[0], 1, a_new[1]])
    Y = ode.solve_ivp(dF, (0, 5), F, max_step=1/500)
    plt.plot(Y.t, Y.y[1], "b", Y.t, Y.y[-2], "r")
    plt.show()

solver(Pr=0.1, error=0.01)
"""


def step(a, it):
    d = 1e-7
    if Pr < 0.04:
        x_0 = 30
        d_it = 15
    elif Pr < 0.07:
        x_0 = 30
        d_it = 2 / (Pr * 10)
    elif Pr < 0.18:
        x_0 = 30
        d_it = 1 / (Pr * 10)
    elif Pr < 0.5:
        x_0 = 20
        d_it = 1 / (Pr * 10)
    elif Pr < 10:
        x_0 = 5
        d_it = 3
    else:
        x_0 = 5
        d_it = 10

    """
    if Pr < 0.1:
        if x_0 + d_it > -50 * np.log10(Pr) - 30:
            x_0 = -50 * np.log10(Pr) - 30
            d_it = 0
    elif Pr < 0:
        if x_0 + d_it > -14 * np.log10(Pr) + 9:
            x_0 = -14 * np.log10(Pr) + 9
            d_it = 0
    else:
        if x_0 + d_it > 5 * np.log10(Pr) + 8.5:
            x_0 = 5 * np.log10(Pr) + 8.5
            d_it = 0
    """

    X, Y = rk4(dF, (0, x_0 + d_it * it), a, h)
    _, Y_a0 = rk4(dF, (0, x_0 + d_it * it), [a[0] + d, a[1]], h)
    __, Y_a1 = rk4(dF, (0, x_0 + d_it * it), [a[0], a[1] + d], h)

    alpha = np.array([Y[-1][1], Y[-1][3]])
    alpha_a0 = [Y_a0[-1][1], Y_a0[-1][3]]
    alpha_a1 = [Y_a1[-1][1], Y_a1[-1][3]]

    J = np.array([
        [(alpha_a0[0] - alpha[0]) / d, (alpha_a1[0] - alpha[0]) / d],
        [(alpha_a0[1] - alpha[1]) / d, (alpha_a1[1] - alpha[1]) / d]
    ])
    da = find_da(J, alpha)
    return a + da


error = 1e-7
h = 0.003


def solver(error=1, N_max=50):
    start = time.process_time()
    # a = [2, -0.6520393]
    # a = [1, -1]
    # a = [(0.295272089 - 0.25169215466) / (50 ** -0.17 - 100 ** -0.17) * (Pr ** -0.17 - 100 ** -0.17) + 0.2516921546, - 2.1913686 * (Pr / 100) ** 0.27]
    a = [- m_0 * Pr ** k + p_0 - 0.034 * np.sin(np.log10(Pr) * np.pi / 2), - m_1 * Pr ** l + p_1 - - 0.02 * np.sin(np.log10(Pr) * np.pi / 2)]
    if 10 < Pr < 90:
        a[1] -= 0.01
    a_new = step(a, 0)
    it = 0
    while (abs(a[0] - a_new[0]) > error or abs(a[1] - a_new[1]) > error) and it <= N_max:
        a = np.copy(a_new)
        a_new = step(a, it)
        it += 1
        print(a_new, " itération numéro", it)
        print(a[0] - a_new[0], a[1] - a_new[1])
    if it <= N_max:
        print(a_new)
    else:
        print("bah c'est baisé...")
    # X, Y = rk4(dF, (0, 50), a_new, h)
    # plt.plot(X, Y[:, 1], "b", X, Y[:, 3], "r")
    # plt.show()
    # plt.plot(X, Y)
    # plt.show()
    print(time.process_time() - start)
    return a_new

def convection(Prandtl, error=1e-7):
    global Pr
    Pr = Prandtl
    a = solver(error)
    return rk4(dF, (0, 50), a, h)

if __name__ == "__main__":
    X, Y = convection(100)


