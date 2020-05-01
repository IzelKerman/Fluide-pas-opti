import numpy as np
import matplotlib.pyplot as plt

import math

from vivelesmaths import *

Pr = 1

def couche_lim(Pr) :
    Gr = 9.81*10000**3/(Pr**3)
    result = 10000*(4/Gr)**0.25
    return result

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

def step(a, it):
    d = 1e-7

    X, Y = rk4(dF, (0, 15 + it / (Pr * 10)), a, h)
    _, Y_a0 = rk4(dF, (0, 15 + it / (Pr * 10)), [a[0] + d, a[1]], h)
    __, Y_a1 = rk4(dF, (0, 15 + it / (Pr * 10)), [a[0], a[1] + d], h)

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
    # a = [2, -0.6520393]
    # a = [1, -1]
    a = [(0.295272089 - 0.25169215466) / (50 ** -0.17 - 100 ** -0.17) * (Pr ** -0.17 - 100 ** -0.17) + 0.2516921546, - 2.1913686 * (Pr / 100) ** 0.27]
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
    # X, Y = rk4(dF, (0, 100), a_new, h)
    # plt.plot(X, Y[:, 1], "b", X, Y[:, 3], "r")
    # plt.show()
    # plt.plot(X, Y)
    return a_new

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

X, Y = rk4(dF, (0, 50), solver(error=1e-7), h)

ax1.plot(X,Y[:,1],"red",label = "f'")
ax1.plot(X, Y[:,3], "blue", label = "theta")

ax1.legend()
ax1.set_ylim(-0.3, 1.3)
ax1.set_xlim(-3, 53)

fig2 = plt.figure("champ de température et de vitesse")
ax2 = fig2.add_subplot(111)

x = np.arange(0, 5, 0.1)
y = np.arange(0.1, 5, 0.1)
Z = np.zeros((len(y), len(x)))
for i, x_i in enumerate(x):
    for j, y_j in enumerate(y):
        if x_i/y_j**(0.25) <= X[-1]:
            k = math.floor(x_i/y_j**(0.25) / h)
            Z[j,i] = Y[k, 3]
        else:
            Z[j,i] = 0
ax2.contourf(x, y, Z, 50, cmap='plasma')

xx_ = np.arange(0.5, 5, 0.5)
yy_ = np.arange(0.5, 5, 0.5)
xx, yy = np.meshgrid(xx_, yy_)
u = np.zeros((len(yy_), len(xx_)))
v = np.zeros((len(yy_), len(xx_)))
for i, x_i in enumerate(xx_):
    for j, y_j in enumerate(yy_):
        if x_i/y_j**(0.25) <= X[-1]:
            k = math.floor(x_i/y_j**(0.25) / h)
            u[j, i] = 1e-1 * (Y[k, 1] * x_i - 3 * y_j ** 0.5 * Y[k, 0] * 1.56e-2) / y_j**0.5
            v[j, i] = 1e-1 * 2 * Y[k, 1] * y_j**0.5
        else:
            u[j, i] = 0 ; v[j, i] = 0
ax2.quiver(xx, yy, u, v,label = "vitesse")

values = []
for i in range(100) :
    if i == 0 :
        values += [couche_lim(0.01)]
    else :
        values += [couche_lim(i)]
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.plot(range(100), values)
