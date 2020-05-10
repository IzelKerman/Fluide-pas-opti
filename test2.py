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
#matplotlib.rcParams['text.usetex'] = True

# données-------------------------------------

m_0 = 3.003451249261589
p_0 = 3.645639392529623
k = 0.026542410398766054
m_1 = 0.6946390572490012
l = 0.2617581820549962
p_1 = 0.1274926083122092

# --------------------------------------------

Pr = 0.1

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


if __name__ == "__main__":
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

    fig = plt.figure("coucou c'est zoli les couleurs")
    ax = fig.add_subplot(1, 1, 1)

    X, Y = rk4(dF, (0, 50), solver(error=1e-7), h)

    points_T = np.array([X, Y[:, 3]]).T.reshape(-1, 1, 2)
    points_v = np.array([X, Y[:, 1]]).T.reshape(-1, 1, 2)
    segments_T = np.concatenate([points_T[:-2], points_T[2:]], axis=1)
    segments_v = np.concatenate([points_v[:-2], points_v[2:]], axis=1)

    T_min = Y[:, 3].min()
    T_max = Y[:, 3].max()
    norm_T = plt.Normalize(T_min, T_max)

    v_min = Y[:, 1].min()
    v_max = Y[:, 1].max()
    norm_v = plt.Normalize(v_min, v_max)

    lc_T = LineCollection(segments_T, cmap='plasma', norm=norm_T)
    lc_v = LineCollection(segments_v, cmap='winter', norm=norm_v)

    lc_T.set_array((Y[-2, 3] + Y[2:, 3]))
    lc_v.set_array((Y[-2, 1] + Y[2:, 1]))
    lc_T.set_linewidth(2)
    lc_v.set_linewidth(2)
    lc_T.set_antialiased(True)
    lc_v.set_antialiased(True)
    line_T = ax.add_collection(lc_T)
    line_v = ax.add_collection(lc_v)

    plt.plot([1, 1], [-0.1, 1.1], '--r')

    plt.ylim(-0.1, 1.1)
    plt.xlim(-3, 23)

    """
    y = np.linspace(1, 5, 20)
    X_x = []
    for y_i in y:
        X_x.append(np.array(X[:-10000])*(y_i**0.25))
    X_y = [np.array([y_j for x_i in X[:-10000]]) for y_j in y]
    print(X_x, X_y)
    print(len(X_x), len(X_y))
    
    for i, y_i in enumerate(X_y):
        plt.scatter(X_x[i], y_i, c=Y[:-10000, 3], cmap="plasma")
    
    
    plt.show()
    """

    fig = plt.figure("test nul")
    plt.plot(X, Y[:, 0], label="$f$")
    plt.plot(X, Y[:, 1], label="$f'$")
    plt.plot(X, Y[:, 2], label="$f''$")
    plt.plot(X, Y[:, 3], label="$\\theta$")
    plt.plot(X, Y[:, 4], label="$\\theta'$")
    plt.legend()

    f_c = 0
    for i, f_i in enumerate(Y[1:, 0]):
        if f_i - Y[i, 0] < 1e-7 and f_i > 0.2:
            f_c = X[i - 1]
            print("got it")
            break
    plt.plot([f_c, f_c], [-1, 1], '--r')

    nu = 15.6e-5
    bgt = 1e-2

    fig = plt.figure("Champs de vitesse et température (Pr = {:.2f})".format(Pr))
    plt.subplot(221)
    plt.title("Champs de vitesse et température calculés (Pr = {:.2f})".format(Pr))

    y_f = 10
    wanted_eta = 4
    x_f = wanted_eta * (4 * nu ** 2 * y_f / bgt) ** 0.25
    # x = np.arange(0, 5, 0.1)
    x = np.linspace(0, 2 * x_f, 2 * 200)
    y = np.linspace(0.1, y_f, 200)
    Z = np.zeros((len(y), len(x)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            if x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                Z[j, i] = Y[k, 3]
                if Z[j, i] < 1e-10:
                    Z[j, i] = 0
            else:
                print("hello there")
                Z[j, i] = 0
    plt.scatter(x[-100], y[-100], zorder=1)
    temperature = plt.contourf(x, y, Z, 200, cmap='plasma')
    cbar = plt.colorbar(temperature)
    cbar.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    n = 20
    # xx_ = np.arange(0.5, 5, 0.5)
    xx_ = np.linspace(x_f / (n + 1), 2 * x_f, 2 * n)
    yy_ = np.linspace(0.5, y_f, n)
    xx, yy = np.meshgrid(xx_, yy_)
    u = np.zeros((len(yy_), len(xx_)))
    v = np.zeros((len(yy_), len(xx_)))
    for i, x_i in enumerate(xx_):
        for j, y_j in enumerate(yy_):
            if x_i * (bgt / (4 * nu ** 2 * y_j)) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                # u[j, i] = bgt**0.5 * (Y[k, 1] * x_i - 3 * y_j ** 0.5 * Y[k, 0] * 1.56e-2) / y_j**0.5
                # u[j, i] = (nu / y_j) * (bgt * y_j ** 3 / (4 * nu**2)) ** 0.25 * (Y[k, 1] * X[k] - 3 * Y[k, 0])
                u[j, i] = 0.5 * (bgt / y_j) ** 0.5 * (Y[k, 1] * x_i - 3 * Y[k, 0] * (4 * nu ** 2 * y_j / bgt) ** 0.25)
                # v[j, i] = bgt**0.5 * 2 * Y[k, 1] * y_j**0.5
                v[j, i] = 2 * (bgt * y_j) ** 0.5 * Y[k, 1]
            else:
                u[j, i] = 1e-5
                v[j, i] = 1e-5

    C = np.sqrt(u ** 2 + v ** 2)
    Norm = plt.Normalize(C.min(), C.max())
    plt.quiver(xx, yy, u, v, C, label="velocity field", cmap='gray_r', norm=Norm, width=0.003, scale=1 / 0.15)

    plt.plot((4 * nu ** 2 * y / bgt) ** 0.25, y, '-r', zorder=1)

    y_0 = 5
    v = np.array([y_0 + 5 * 2 * (bgt * y_0) ** 0.5 * Y[math.floor(x_i * (bgt / (4 * nu ** 2 * (y_0))) ** (0.25) / h), 1] for i, x_i in enumerate(x)])
    max_v = np.max(v - y_0)
    plt.plot(x, v)

    plt.subplot(222)
    plt.title("Champs de vitesse et température approximés (Pr = {:.2f})".format(Pr))

    #D = (240 * (20/21 + Pr) * nu ** 2 / bgt / Pr ** 2)**0.25
    D = (240 * ((20 / 21) + Pr) * nu ** 2 / (bgt * (Pr ** 2))) ** 0.25
    V = 80 * nu / (Pr * (D ** 2))

    m_p = 0.5
    n_p = 0.25

    def v_0(y):
        return V * y ** m_p

    def delta(y):
        return D * y ** n_p

    Z_ = np.zeros((len(y), len(x)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            if delta(y_j) > x_i:
                Z_[j, i] = (1 - x_i / delta(y_j)) ** 2
            else:
                Z_[j, i] = 0
    temperature = plt.contourf(x, y, Z_, 200, cmap='plasma')
    cbar = plt.colorbar(temperature)
    cbar.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    y_0 = 5
    vy_0 = []
    for x_i in x:
        if delta(y_0) > x_i:
            vy_0.append(y_0 + 5 * v_0(y_0) * ( x_i / delta(y_0)) * (1 - x_i / delta(y_0)) ** 2)
        else:
            vy_0.append(y_0)
    vy_0 = np.array(vy_0)
    max_vy_0 = np.max(vy_0 - y_0)
    plt.plot(x, vy_0)

    plt.subplot(223)
    plt.title("Différence absolue des modèles de température (Pr = {:.2f})".format(Pr))

    temperature = plt.contourf(x, y, np.abs(Z - Z_), 200, cmap='plasma')
    cbar = plt.colorbar(temperature)
    cbar.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    plt.subplot(224)
    plt.title("Différence des modèles de vitesse (Pr = {:.2f})".format(Pr))

    plt.plot(x, (v - vy_0) / max(max_v, max_vy_0))
    #plt.plot(x, (v - y_0) / max(max_v, max_vy_0))
    #plt.plot(x, (vy_0 - y_0) / max(max_v, max_vy_0))
    plt.ylim(-1.1, 1.1)



    fig = plt.figure("Champs de vitesse et température nu et alpha cste (Pr = {:.2f})".format(Pr))
    plt.subplot(121)
    plt.title("nu fixe, Pr={:.2f}".format(Pr))

    y_f = 10
    wanted_eta = 4
    x_f = wanted_eta * (4 * nu ** 2 * y_f / bgt) ** 0.25
    # x = np.arange(0, 5, 0.1)
    x = np.linspace(0, 2 * x_f, 2 * 200)
    y = np.linspace(0.1, y_f, 200)
    Z = np.zeros((len(y), len(x)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            if x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                Z[j, i] = Y[k, 3]
                if Z[j, i] < 1e-10:
                    Z[j, i] = 0
            else:
                print("hello there")
                Z[j, i] = 0
    plt.scatter(x[-100], y[-100], zorder=1)
    temperature = plt.contourf(x, y, Z, 200, cmap='plasma')
    cbar = plt.colorbar(temperature)
    cbar.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    n = 20
    # xx_ = np.arange(0.5, 5, 0.5)
    xx_ = np.linspace(x_f / (n + 1), 2 * x_f, 2 * n)
    yy_ = np.linspace(0.5, y_f, n)
    xx, yy = np.meshgrid(xx_, yy_)
    u = np.zeros((len(yy_), len(xx_)))
    v = np.zeros((len(yy_), len(xx_)))
    for i, x_i in enumerate(xx_):
        for j, y_j in enumerate(yy_):
            if x_i * (bgt / (4 * nu ** 2 * y_j)) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                # u[j, i] = bgt**0.5 * (Y[k, 1] * x_i - 3 * y_j ** 0.5 * Y[k, 0] * 1.56e-2) / y_j**0.5
                # u[j, i] = (nu / y_j) * (bgt * y_j ** 3 / (4 * nu**2)) ** 0.25 * (Y[k, 1] * X[k] - 3 * Y[k, 0])
                u[j, i] = 0.5 * (bgt / y_j) ** 0.5 * (Y[k, 1] * x_i - 3 * Y[k, 0] * (4 * nu ** 2 * y_j / bgt) ** 0.25)
                # v[j, i] = bgt**0.5 * 2 * Y[k, 1] * y_j**0.5
                v[j, i] = 2 * (bgt * y_j) ** 0.5 * Y[k, 1]
            else:
                u[j, i] = 1e-5
                v[j, i] = 1e-5

    C = np.sqrt(u ** 2 + v ** 2)
    Norm = plt.Normalize(C.min(), C.max())
    plt.quiver(xx, yy, u, v, C, label="velocity field", cmap='gray_r', norm=Norm, width=0.003, scale=1 / 0.15)

    plt.plot((4 * nu ** 2 * y / bgt) ** 0.25, y, '-r', zorder=1)

    y_0 = 5
    v = np.array([y_0 + 5 * 2 * (bgt * y_0) ** 0.5 * Y[math.floor(x_i * (bgt / (4 * nu ** 2 * (y_0))) ** (0.25) / h), 1] for i, x_i in enumerate(x)])
    max_v = np.max(v - y_0)
    plt.plot(x, v)


    plt.subplot(122)
    plt.title("alpha fixe, Pr={:.2f}".format(Pr))

    alpha = 19
    nu = Pr/alpha
    y_f = 10
    wanted_eta = 4
    x_f = wanted_eta * (4 * nu ** 2 * y_f / bgt) ** 0.25
    # x = np.arange(0, 5, 0.1)
    x = np.linspace(0, 2 * x_f, 2 * 200)
    y = np.linspace(0.1, y_f, 200)
    Z = np.zeros((len(y), len(x)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            if x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                Z[j, i] = Y[k, 3]
                if Z[j, i] < 1e-10:
                    Z[j, i] = 0
            else:
                print("hello there")
                Z[j, i] = 0
    plt.scatter(x[-100], y[-100], zorder=1)
    temperature = plt.contourf(x, y, Z, 200, cmap='plasma')
    cbar = plt.colorbar(temperature)
    cbar.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    n = 20
    # xx_ = np.arange(0.5, 5, 0.5)
    xx_ = np.linspace(x_f / (n + 1), 2 * x_f, 2 * n)
    yy_ = np.linspace(0.5, y_f, n)
    xx, yy = np.meshgrid(xx_, yy_)
    u = np.zeros((len(yy_), len(xx_)))
    v = np.zeros((len(yy_), len(xx_)))
    for i, x_i in enumerate(xx_):
        for j, y_j in enumerate(yy_):
            if x_i * (bgt / (4 * nu ** 2 * y_j)) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                # u[j, i] = bgt**0.5 * (Y[k, 1] * x_i - 3 * y_j ** 0.5 * Y[k, 0] * 1.56e-2) / y_j**0.5
                # u[j, i] = (nu / y_j) * (bgt * y_j ** 3 / (4 * nu**2)) ** 0.25 * (Y[k, 1] * X[k] - 3 * Y[k, 0])
                u[j, i] = 0.5 * (bgt / y_j) ** 0.5 * (Y[k, 1] * x_i - 3 * Y[k, 0] * (4 * nu ** 2 * y_j / bgt) ** 0.25)
                # v[j, i] = bgt**0.5 * 2 * Y[k, 1] * y_j**0.5
                v[j, i] = 2 * (bgt * y_j) ** 0.5 * Y[k, 1]
            else:
                u[j, i] = 1e-5
                v[j, i] = 1e-5

    C = np.sqrt(u ** 2 + v ** 2)
    Norm = plt.Normalize(C.min(), C.max())
    plt.quiver(xx, yy, u, v, C, label="velocity field", cmap='gray_r', norm=Norm, width=0.003, scale=1 / 0.15)

    plt.plot((4 * nu ** 2 * y / bgt) ** 0.25, y, '-r', zorder=1)

    y_0 = 5
    v = np.array([y_0 + 5 * 2 * (bgt * y_0) ** 0.5 * Y[math.floor(x_i * (bgt / (4 * nu ** 2 * (y_0))) ** (0.25) / h), 1] for i, x_i in enumerate(x)])
    max_v = np.max(v - y_0)
    plt.plot(x, v)

    plt.show()





