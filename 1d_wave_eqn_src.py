import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

L, T, c, C = 2, 10, 2, 0.25
Nx = 300
dx = L/Nx
dt = (C*dx)/c
Nt = int(round(T/dt))

t = np.linspace(0, T, Nt)
x = np.linspace(0, L, Nx)

I = lambda x: 0
f = lambda x,t: np.sin(30*np.pi*t/L)

u 		= np.zeros(Nx)
u_n		= np.zeros(Nx)
u_nm1	= np.zeros(Nx)

bc = 'nc'
U = np.zeros((Nt,Nx))

#for i in range(Nx): u_n[i] = I(x[i])
u_n[int(Nx/2)] = np.sin(10*np.pi*t[0]) 

u[1:-1] = u_n[1:-1] + 0.5 * C**2 * (u_n[2:] - 2*u_n[1:-1] + u_n[:-2])

if bc=="dc":
	# Dirichlet Boundary Condion
	u[0], u[-1] = 0, 0 
else:
	# Neumann Boundary Condion
	u[0]  = u_n[0] + C**2 * (u_n[1] - u_n[0])
	u[-1] = u_n[-1] + C**2 * (u_n[-2] - u_n[-1])
u_nm1[:], u_n[:] = u_n, u
U[0,:] = u

for n in range(1,Nt):
	u_n[int(Nx/2)] = np.sin(10*np.pi*t[n])
	u[1:-1] = 2*u_n[1:-1] - u_nm1[1:-1] + C**2 * (u_n[2:] - 2*u_n[1:-1] + u_n[:-2]) + f(1.5,t[n])*dt**2
	if bc=="dc":
		# Dirichlet Boundary Condion
		u[0], u[-1] = 0, 0
	else:
		# Neumann Boundary Condion
		u[0] = 2*u_n[0] - u_nm1[0] + C**2 * (u_n[1] - u_n[0])
		u[-1] = 2*u_n[-1] - u_nm1[-1] + C**2 * (u_n[-2] - u_n[-1])
	u_nm1[:], u_n[:] = u_n, u
	U[n,:] = u

fig = plt.figure()
axis = plt.axes(xlim=(0, L), ylim=(-1.5,1.5))
line, = plt.plot([],[], lw=2, color='red')
plt.grid()

def animate(i):
	plt.title(f"t = {t[i]:.4f}")
	line.set_data(x,U[i])
	return line,

anim = FuncAnimation(fig, animate, interval = 5, frames=Nt, repeat=True)
#anim.save("1d_wave_packet.mp4",writer="ffmpeg")
plt.show()