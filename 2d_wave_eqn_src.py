import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Lx, Ly, T, c, C = 2, 2, 1, 2, 0.24
Nx = 250
dh = Lx/Nx
Ny = int(round(Ly/dh))
dt = (C*dh)/c 
Nt = int(round(T/dt))
Cm = (C-1)/(C+1)

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
t = np.linspace(0, T, Nt)


f1 = lambda t: np.sin(30*np.pi*t) 
f2 = lambda t: np.sin(30*np.pi*t) 

u 		= np.zeros((Nx,Ny))
u_n 	= np.zeros((Nx,Ny))
u_nm1 	= np.zeros((Nx,Ny))
U 		= np.zeros((Nt,Nx,Ny))

X,Y = np.meshgrid(x,y)

#At t=0
u_n[int(Nx/2),int(Ny/2 + Ny/8)] = f1(t[0])
u_n[int(Nx/2),int(Ny/2 - Ny/8)] = f2(t[0])

u_xx = u_n[2:,1:-1] - 2*u_n[1:-1,1:-1] + u_n[:-2,1:-1]
u_yy = u_n[1:-1,2:] - 2*u_n[1:-1,1:-1] + u_n[1:-1,:-2]
u[1:-1,1:-1] = u_n[1:-1,1:-1] + 0.5 * C**2 * (u_xx + u_yy)

#Mur's Absorbing boundary conndition

u[0,:] = u_n[1,:] + Cm * (u_n[1,:] - u_n[0,:])
u[-1,:] = u_n[-2,:] + Cm * (u_n[-2,:] - u_n[-1,:])
u[:,0] = u_n[:,1] + Cm * (u_n[:,1] - u_n[:,0])
u[:,-1] = u_n[:,-2] + Cm * (u_n[:,-2] - u_n[:,-1])
u_nm1[:,:], u_n[:,:] = u_n, u
U[0,:,:] = u

# For t>0
for n in range(1, Nt):
	u_n[int(Nx/2),int(Ny/2 + Ny/8)] = f1(t[n])
	u_n[int(Nx/2),int(Ny/2 - Ny/8)] = f2(t[n])

	u_xx = u_n[2:,1:-1] - 2*u_n[1:-1,1:-1] + u_n[:-2,1:-1]
	u_yy = u_n[1:-1,2:] - 2*u_n[1:-1,1:-1] + u_n[1:-1,:-2]
	u[1:-1,1:-1] = 2*u_n[1:-1,1:-1] - u_nm1[1:-1,1:-1] + C**2 * (u_xx + u_yy)
	
	#Mur's Absorbing boundary conndition
	u[0,:] = u_n[1,:] + Cm * (u_n[1,:] - u_n[0,:])
	u[-1,:] = u_n[-2,:] + Cm * (u_n[-2,:] - u_n[-1,:])
	u[:,0] = u_n[:,1] + Cm * (u_n[:,1] - u_n[:,0])
	u[:,-1] = u_n[:,-2] + Cm * (u_n[:,-2] - u_n[:,-1])

	u_nm1[:,:], u_n[:,:] = u_n, u
	U[n,:,:] = u

"""
I = abs(U[960,100,:])**2
plt.plot(x,I)
"""

fig = plt.figure()
def animate(i):
	plt.clf()
	plt.title(f"t = {t[i]:.2f} frame = {i}")
	plt.imshow(U[i],vmax=0.2,vmin=-0.2,cmap=plt.cm.jet, interpolation='gaussian')
	plt.colorbar()
	return plt

anim = FuncAnimation(fig, animate, interval=10, frames=Nt, repeat=True)
#anim.save("2d_wave.mp4",writer="ffmpeg")
plt.show()
