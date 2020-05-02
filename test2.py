import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.collections import LineCollection
import scipy.integrate as ode

import math

from vivelesmaths import *

import multiprocessing as mp

from multiprocessing import Process

import csv
import os

import time

plt.style.use('dark_background')

#données-------------------------------------

m_0 = 3.003451249261589
p_0 = 3.645639392529623
k = 0.026542410398766054
m_1 = 0.6946390572490012
l = 0.2617581820549962
p_1 = 0.1274926083122092

#--------------------------------------------

Pr = 100

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

def rk4_2(F):
    dF = F[0]; x_0 = F[1]; a = F[2]; h = F[3]
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
    return [X, U]


def find_da(J, alpha):
    """
    for i in range(2):
        if J[i][0] < 1e-10 and J[i][1] < 1e-10:
            j = (i+1) % 2
            if J[j][0] < 1e-10 and J[j][1] < 1e-10:
                return np.zeros(2)
            else:
                grad = np.array(J[j])
                print("grad = ", grad)
                dist = alpha[j]/np.linalg.norm(grad)
                da = - dist * grad / np.linalg.norm(grad)
                print("dist = ", dist)
                return da
    else:
        return Gauss(-J, alpha)
    """
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

    if Pr < 0.5:
        x_0 = 60
        d_it = 1 / (Pr * 10)
    elif Pr < 10:
        x_0 = 15
        d_it = 1
    elif Pr < 90:
        x_0 = 5
        d_it = 10
    else:
        x_0 = 10
        d_it = 10

    with mp.Pool(3) as p:
        S = p.map(rk4_2, [[dF, (0, x_0 + d_it * it), a, h], [dF, (0, x_0 + d_it * it), [a[0] + d, a[1]], h], [dF, (0, x_0 + d_it * it), [a[0], a[1] + d], h]])

    alpha = np.array([S[0][1][-1][1], S[0][1][-1][3]])
    alpha_a0 = [S[1][1][-1][1], S[1][1][-1][3]]
    alpha_a1 = [S[2][1][-1][1], S[2][1][-1][3]]

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
    if it <= N_max:
        print(a_new)
    else:
        print("bah c'est baisé...")
    X, Y = rk4(dF, (0, 50), a_new, h)
    # plt.plot(X, Y[:, 1], "b", X, Y[:, 3], "r")
    # plt.show()
    # plt.plot(X, Y)
    # plt.show()
    print(time.process_time() - start)
    return a_new

"""
P = [(i+1)*1e-2 for i in range(9)] + [(i+1)*1e-1 for i in range(9)] + [(i+1) for i in range(9)] + [(i+1)*1e1 for i in range(10)]
A = []
for i in range(len(P)):
    Pr = P[i]
    A.append(solver(error=1e-7))
    print("{:.2f}%".format((1+i)/len(P)*100))


# Ecriture des donnée dans un fichier
with open("Pr_data_better.csv", 'a', newline='') as file:
    if os.stat("Pr_data_better.csv").st_size == 0:
        writer = csv.writer(file)
        writer.writerow(["Pr", "a_0", "a_1"])
        for i in range(len(P)):
            writer.writerow([P[i], A[i][0], A[i][1]])
"""

if __name__ == '__main__':
    start = time.time()
    X, Y = rk4(dF, (0, 50), solver(error=1e-7), h)
    print("total time = ", time.time() - start)

    fig = plt.figure("Coucou")
    plt.plot(X, Y[:, 1], "-b", X, Y[:, 3], "-r")
    plt.show()


