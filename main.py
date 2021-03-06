import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import time

# plt.style.use('dark_background')
# plt.rc('text', usetex=True)

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

def step(a, it):
    d = 1e-7
    x_0 = 10
    d_it = 10

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
    da = np.linalg.solve(-J, alpha)
    return a + da


error = 1e-7
h = 0.003

def guess_initial_conditions():
    Pr_reference = np.array([0.01, 0.1, 1, 10, 100])
    fpp_reference = np.array([0.98755568, 0.8591655,  0.64223382, 0.41933443, 0.25169789])
    thetap_reference = np.array([-0.0806193, -0.23015915, -0.56726326, -1.16969874, -2.1914066])
    # interpolation en log10(Pr) par splines cubiques
    fpp_interp = interp1d(np.log10(Pr_reference), fpp_reference, kind='cubic')
    thetap_interp = interp1d(np.log10(Pr_reference), thetap_reference, kind='cubic')
    return np.array([fpp_interp(np.log10(Pr)), thetap_interp(np.log10(Pr))])

def solver(error=1, N_max=50):
    start = time.process_time()
    a = guess_initial_conditions()
    a_new = step(a, 0)
    it = 0
    while (abs(a[0] - a_new[0]) > error or abs(a[1] - a_new[1]) > error) and it <= N_max:
        a = np.copy(a_new)
        a_new = step(a, it)
        it += 1
        print(a_new, "itération numéro", it)
        print(a[0] - a_new[0], a[1] - a_new[1])
    if it <= N_max:
        print(a_new)
    else:
        print("ERREUR: %d itérations n'ont pas suffi :-(" % N_max)
    print(time.process_time() - start)
    return a_new

def convection(Prandtl, error=1e-7, graph=True):
    global Pr
    Pr = Prandtl
    a = solver(error)
    if graph:
        X, Y = rk4(dF, (0, 50), a, h)
        plt.plot(X, Y[:, 1], '-b', label="f'")
        plt.plot(X, Y[:, 3], '-r', label="theta")
        plt.xlabel("eta")
        plt.legend()
        plt.show()
    return a[0], a[1]

if __name__ == "__main__":
    X, Y = convection(10)



