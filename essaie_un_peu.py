import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import scipy.integrate as ode
from multiprocessing import *

from nath2 import *

import time

Pr = 1

def dF(_, F):
    return np.array([
        F[1],
        F[2],
        2 * F[1]**2 - 3 * F[0] * F[2] - F[3],
        F[4],
        - 3 * Pr * F[0] * F[4]
    ])

def rk4(dF, x_0, a, h, index, data):
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
    if index == 1 :
        data[0] = X
    data[index] = U
        

def find_da(J, alpha):
    return Gauss(-J, alpha)



#ode.solve_ivp(dF, (0, 5), Y, max_step=1/500)


def step(a, it):
    d = 1e-7

    F = np.array([0, 0, a[0], 1, a[1]])
    F_a0 = np.array([0, 0, a[0] + d, 1, a[1]])
    F_a1 = np.array([0, 0, a[0], 1, a[1] + d])
    l = Manager().list([0,0,0,0])
    line1 = Process(target = rk4, args = (dF, (0, 15 + it/(Pr * 10)), a, h,1, l))
    line2 = Process(target = rk4, args = (dF, (0, 15 + it/(Pr * 10)), [a[0] + d, a[1]], h, 2, l))
    line3 = Process(target = rk4, args = (dF, (0, 15 + it/(Pr * 10)), [a[0], a[1] + d], h, 3, l))
    line1.start()
    line2.start()
    line3.start()
    line1.join()
    line2.join()
    line3.join()
    X = l[0]
    Y = l[1]
    Y_a0 = l[2]
    Y_a1 = l[3]
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
    #a = [2, -0.6520393]
    #a = [1, -1]
    a = [(0.295272089 - 0.25169215466) / (50**-0.17 - 100**-0.17) * (Pr ** -0.17 - 100**-0.17) + 0.2516921546, - 2.1913686 * (Pr/100) ** 0.27]
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
    print(time.process_time() - start)
    l2 = Manager().list([0,0])
    rk4(dF, (0, 50), a_new, h, 1, l2)
    plt.plot(l2[0], l2[1][:, 1], "b", l2[0], l2[1][:, 3], "r")
    plt.show()
    plt.plot(l2[0], l2[1])
    return a_new


if __name__ == '__main__':
    freeze_support()
    solver(error=1e-7)
