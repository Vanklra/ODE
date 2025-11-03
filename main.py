import numpy as np
from scipy.integrate import odeint, solve_ivp
import matplotlib.pyplot as plt


sinh = np.sinh(np.pi)

def f(t, y):
    global sinh

    Y1 = y[0]
    Y2 = y[1]
    Y3 = y[2]
    Y4 = y[3]
    dydt = [np.pi * Y3 * sinh - 2 * np.pi ** 2 * Y2 * Y4 * np.cosh(np.pi * np.sqrt(2)) - (
                4 * np.pi / 3) * Y1 * Y3 * np.cosh(np.pi),

            np.pi * np.sqrt(2) * Y4 * np.sinh(np.pi * np.sqrt(2)) - (4 * np.pi / 3) * Y1 * Y4 * np.cosh(
                np.pi * np.sqrt(2)) - (4 * np.pi / 3) * Y2 * Y3 * np.cosh(np.pi),


            -np.pi * Y4 * np.cosh(np.pi * np.sqrt(2)) - (2 * np.pi / 3) * Y3 ** 2 * np.cosh(np.pi) - (
                        4 * np.pi / (3 * np.cosh(np.pi))) * Y3 ** 2 * (np.sinh(np.pi)) ** 2 - (
                        4 * np.pi / (3 * np.cosh(np.pi))) * Y4 ** 2 * (np.sinh(np.pi * np.sqrt(2))) ** 2 - (
                        10 * Y1 / np.cosh(np.pi)),


            -(4 * np.pi / (3 * np.cosh(np.pi))) * Y3 * Y4 * sinh - (
                        8 * np.pi * np.sqrt(2) / (3 * np.cosh(np.pi * np.sqrt(2)))) * Y3 * Y4 * np.sinh(
                np.pi) * np.sinh(np.pi * np.sqrt(2)) - (10 * Y2 / np.cosh(np.pi * np.sqrt(2)))]

    return dydt

tspan = np.array((0, 0.8))
ics = [-1, 0.1, 0.01, 0.001]
#y = odeint(f, ics, tspan)

res = solve_ivp(f, tspan, ics)
y = res.y
t = res.t
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
filename = 'plot_1.gif'

rg = np.linspace(0, 1, 100)
X, Y = np.meshgrid(rg, rg)

CosPiX = np.cos(np.pi * X)
SinPiY = np.sin(np.pi * Y)

for i, T in enumerate(t):
    Q01 = y[0,i]
    Q11 = y[1,i]
    Z = 1 + Q01 * SinPiY + Q11 * CosPiX * SinPiY
    ax.clear()
    ax.plot_surface(X, Y, Z, cmap='jet')
    ax.set_title('f(x,y,z) при t = {:.2f}'.format(T))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.savefig('frame_{:03d}.png'.format(i))

fig = plt.figure()
plt.plot(t, y[0], '-o', label='Y1')
plt.plot(t, y[1], '-o', label='Y2')
plt.plot(t, y[2], '-o', label='Y3')
plt.plot(t, y[3], '-o', label='Y4')
plt.xlabel('t')
plt.ylabel('Q/P')
plt.title('Values Y1, Y2, Y3, Y4')
plt.legend()
plt.grid()
plt.show()

a = 0
