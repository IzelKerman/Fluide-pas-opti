from main import *
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import math


#P = [[(i + 1) / 1000, 1] for i in range(200)]
#a = solver(error=1e-7)
#X, Y = rk4(dF, (0, 100), a, h)

nu = 1.56e-5
bgt = 1e-2
dt = 0.001
N = 150

"""
def animation(im):
    global X, Y, nu, bgt, dt

    ax.clear()

    f = 0
    df = 0
    for j in range(1000):
        for p in P:
            eta = p[0] * (bgt / (4 * nu ** 2 * (p[1]))) ** (0.25)
            k = math.floor(eta / h)
            f = (Y[k + 1, 0] - Y[k, 0]) / h * (eta - X[k]) + Y[k, 0]
            df = (Y[k + 1, 1] - Y[k, 1]) / h * (eta - X[k]) + Y[k, 1]
            p[0] += (0.5 * (bgt / p[1]) ** 0.5 * (df * p[0] - 3 * f * (4 * nu ** 2 * p[0] / bgt) ** 0.25)) * dt
            p[1] += 2 * (bgt * p[1]) ** 0.5 * df * dt

    y_f = 10
    wanted_eta = 10
    x_f = wanted_eta * (4 * nu ** 2 * y_f / bgt) ** 0.25
    x = np.linspace(0, x_f, 200)
    y = np.linspace(0.1, y_f, 200)
    Z = np.zeros((len(y), len(x)))
    for i, x_i in enumerate(x):
        for j, y_j in enumerate(y):
            if x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) <= X[-1]:
                k = math.floor(x_i * (bgt / (4 * nu ** 2 * (y_j))) ** (0.25) / h)
                Z[j, i] = Y[k, 3]
            else:
                Z[j, i] = 0
    temperature = plt.contourf(x, y, Z, 200, cmap='plasma')
    #cbar = plt.colorbar(temperature)
    #cbar.ax.set_ylabel('T', rotation=0, fontsize=20, labelpad=10)

    ax.set_xlim(0, x_f)
    ax.set_ylim(0.1, y_f)

    plt.scatter([p[0] for p in P], [p[1] for p in P], c=[k for k in range(len(P))], cmap='viridis', zorder=1)

    print("{:.2f}%".format(im / N * 100))
"""

N = 24 * 15


def animation2(im):
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    Pr = 0.1 + 99.9 * (1 - np.cos(np.pi * (np.exp(np.log(N + 1) * im / N) - 1) / N)) / 2
    if Pr > 100:
        Pr = 100
    X, Y = convection(Pr)
    print("Done at {:.1f}%".format(im / N * 100))
    print("Pr = ", Pr)

    ax1.set_title("Champs de vitesse et température (calcul numérique) (Pr = {:.2f})".format(Pr))

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
                Z[j, i] = 0
    temperature = ax1.contourf(x, y, Z, 200, cmap='plasma')
    #cbar_1 = plt.colorbar(temperature, ax=ax1)
    #cbar_1.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

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
    ax1.quiver(xx, yy, u, v, C, label="velocity field", cmap='gray_r', norm=Norm, width=0.003, scale=1 / 0.15)

    ax1.plot((4 * nu ** 2 * y / bgt) ** 0.25, y, '-r', zorder=1)

    y_0 = 5
    v = np.array([y_0 + 5 * 2 * (bgt * y_0) ** 0.5 * Y[math.floor(x_i * (bgt / (4 * nu ** 2 * (y_0))) ** (0.25) / h), 1] for i, x_i in enumerate(x)])
    max_v = np.max(v - y_0)
    ax1.plot(x, v)

    ax2.set_title("Champs de vitesse et température (approximation)) (Pr = {:.2f})".format(Pr))

    # D = (240 * (20/21 + Pr) * nu ** 2 / bgt / Pr ** 2)**0.25
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
    temperature = ax2.contourf(x, y, Z_, 200, cmap='plasma')
    #cbar_2 = plt.colorbar(temperature)
    #cbar_2.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    y_0 = 5
    vy_0 = []
    for x_i in x:
        if delta(y_0) > x_i:
            vy_0.append(y_0 + 5 * v_0(y_0) * (x_i / delta(y_0)) * (1 - x_i / delta(y_0)) ** 2)
        else:
            vy_0.append(y_0)
    vy_0 = np.array(vy_0)
    max_vy_0 = np.max(vy_0 - y_0)
    ax2.plot(x, vy_0)

    ax3.set_title("Différence absolue des modèles de température (Pr = {:.2f})".format(Pr))

    temperature = ax3.contourf(x, y, np.abs(Z - Z_), 200, cmap='plasma')
    #cbar_3 = plt.colorbar(temperature)
    #cbar_3.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)

    ax4.set_title("Différence des modèles de vitesse (Pr = {:.2f})".format(Pr))

    ax4.plot(x, (v - vy_0) / max(max_v, max_vy_0))
    ax4.set_ylim(-1.1, 1.1)


fig = plt.figure(dpi=100, figsize=(12.8 * 1.5, 9.6 * 1.5))
ax1 = fig.add_subplot(221)
cbar_1 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.plasma), ax=ax1)
cbar_1.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)
ax2 = fig.add_subplot(222)
cbar_2 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=plt.cm.plasma), ax=ax2)
cbar_2.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)
ax3 = fig.add_subplot(223)
cbar_3 = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 0.7), cmap=plt.cm.plasma), ax=ax3)
cbar_3.ax.set_ylabel('$\\theta$', rotation=0, fontsize=20, labelpad=10)
ax4 = fig.add_subplot(224)

Writer = anim.writers['ffmpeg']
writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=-1)

ani = anim.FuncAnimation(fig, animation2, frames=N + 1)
ani.save('yolo.mp4', writer=writer)
plt.show()
