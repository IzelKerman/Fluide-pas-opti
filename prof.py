from matplotlib import pyplot as plt
import numpy as np

Pr = 1

def df(F):
    return np.array([F[1],F[2], 2 * F[1] ** 2 - 3 * F[0] * F[2] - F[3], F[4], -3*F[0]*Pr*F[4]])
 
def rk4(a,h, N=7):
    imax = int(N/h)
    X = np.arange(imax+1)*h
    U = np.zeros((imax+1,5)); U[0,:] = [0,0,a[0],1,a[1]]
    for i in range(imax):  
        K1 = df(U[i,:]       )
        if K1[2] > 1e5 : #mon mécanisme anti-dégénérécence qui marche pas
            U[-1,:] = np.array([1000,1000,1000,1000,1000])
            break
        K2 = df(U[i,:]+K1*h/2)
        K3 = df(U[i,:]+K2*h/2)
        K4 = df(U[i,:]+K3*h  )
        U[i+1,:] = U[i,:] + h*(K1+2*K2+2*K3+K4)/6
        
    return X,U

def shoot(a,h):
    X,U = rk4(a,h)
    return U[-1,1], U[-1,3], U[-1,2],U[-1,4]
  

a_initial = [(0.295272089 - 0.25169215466) / (50 ** -0.17 - 100 ** -0.17) * (Pr ** -0.17 - 100 ** -0.17) + 0.2516921546, - 2.1913686 * (Pr / 100) ** 0.27]  

def blasius(h,tol, init): #elle marche pas trop
    n = 1; nmax = 50;
    a1 = init[0]-0.01; b1 = init[0]+0.01
    a2 = init[1]-0.01; b2 = init[1]+0.01
    fa1, fa2, dfa1, dfa2 = shoot([a1,a2],h); fb1, _, dfb1,__ = shoot([b1,a2],h)
    _, fb2, __, dfb2 = shoot([a1,b2],h)
    delta1 = (b1-a1)/2; delta2 = (b2-a2)/2
    while ((abs(delta1) >= tol or abs(delta2) >= tol) and n <= nmax) :
        delta1 = (b1-a1)/2; delta2 = (b2-a2)/2; n += 1
        x = a1 + delta1; fx,_, dfx,__ = shoot([x,a2],h)
        y = a2 + delta2; _,fy,__,dfy = shoot([a1,y],h)
        if (fx < fa1 and dfx > 0 and dfx < dfa1) : #ici, je sélectionne la meilleure intervalle, donc au dessus ou en dessous
            a1 = x;  fa1 = fx; dfa1 = dfx
        else :
            b1 = x;  fb1 = fx; dfb1 = dfx
        if (fy > fa2 and dfy > 0 and dfy < dfa2) :
            a2 = y;  fa2 = fy; dfa2 = dfy
        else :
            b2 = y;  fb2 = fy; dfb2 = dfy
    if (n > nmax) :
      print("c'est la baise")
    return [x,y]


def targeting(h, tol, init,dist) : #marche que pour 1 et 2
    a1 = init[0]-dist
    a2 = init[1]-dist
    minimum = 10000
    values = [0,0]
    step = 2*dist/10
    while step > tol :
        for i in range(11) :
            for j in range(11) :
                A, B,C,D = shoot([a1 + i*step,a2 + j*step],h)
                if minimum > A+B and A > 0 and B > 0:
                    minimum = A+B
                    values = [a1 + i*step,a2 + j*step]
        step /= 10
        a1 = values[0] - 5*step
        a2 = values[1] - 5*step
    return values

h   = 0.1
tol = 1e-7
a = targeting(h,tol, a_initial, 0.1)
X,U = rk4(a,h)
fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)

ax1.plot(X,U[:,1],"red",label = "f'")
ax1.plot(X, U[:,3], "blue", label = "theta")

ax1.legend()
ax1.set_xlabel("eta")
ax1.axvline(x = 1)