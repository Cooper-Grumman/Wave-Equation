import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

Lx, Ly, T, c, C = 3, 3, 14, 2, 0.24
Nx = 150
dh = Lx/Nx
Ny = int(round(Ly/dh))
dt = (C*dh)/c 
Nt = int(round(T/dt))

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
t = np.linspace(0, T, Nt)

I = lambda x,y: np.exp(-((X-1.5)**2 + (Y-1.5)**2)/0.05)

u 		= np.zeros((Nx,Ny))
u_n 	= np.zeros((Nx,Ny))
u_nm1 	= np.zeros((Nx,Ny))
U 		= np.zeros((Nt,Nx,Ny))

X,Y = np.meshgrid(x,y)
u_n = I(X,Y)

#At t=0
u_n_xx = u_n[2:,1:-1] - 2*u_n[1:-1,1:-1] + u_n[:-2,1:-1]
u_n_yy = u_n[1:-1,2:] - 2*u_n[1:-1,1:-1] + u_n[1:-1,:-2]
u[1:-1,1:-1] = u_n[1:-1,1:-1] + 0.5 * C**2 * (u_n_xx + u_n_yy)

#Dirichlet Condition
u[:,0], u[:,-1], u[0,:], u[-1,:] = 0, 0, 0, 0
u_nm1[:,:], u_n[:,:] = u_n, u
U[0,:,:] = u

#t > 0
for n in range(1, Nt):
	u_n_xx = u_n[2:,1:-1] - 2*u_n[1:-1,1:-1] + u_n[:-2,1:-1]
	u_n_yy = u_n[1:-1,2:] - 2*u_n[1:-1,1:-1] + u_n[1:-1,:-2]
	u[1:-1,1:-1] = 2*u_n[1:-1,1:-1] - u_nm1[1:-1,1:-1] + C**2 * (u_n_xx + u_n_yy)
	
	#Dirichlet Condition
	u[:,0], u[:,-1], u[0,:], u[-1,:] = 0, 0, 0, 0
	u_nm1[:,:], u_n[:,:] = u_n, u
	U[n,:,:] = u

"""
fig = plt.figure()
def animate(i):
	plt.clf()
	plt.title(f"t = {t[i]:.4f}")
	ax = plt.axes(projection='3d')
	ax.set_xlim(0,3)
	ax.set_ylim(0,3)
	ax.set_zlim(-1,1)
	p = ax.plot_surface(X,Y,U[i], linewidth=0.01,vmax=0.2,vmin=-0.2,
		cmap=plt.cm.jet,antialiased=True)
	fig.colorbar(p, ax=ax, pad=0.06, shrink=0.8)
	return ax
	p = plt.imshow(U[i],vmax=0.2,vmin=-0.2,cmap=plt.cm.jet)
	plt.colorbar()
	return plt
"""

fig = plt.figure(figsize=(5,10))
def animate2(i):
	plt.clf()

	ax1 = fig.add_subplot(211, projection='3d')
	ax1.set_title(f"t = {t[i]:.4f}")
	ax1.set_xlim(0,3)
	ax1.set_ylim(0,3) 
	ax1.set_zlim(-1,1)
	p1 = ax1.plot_surface(X,Y,U[i],alpha=1,linewidth=0.01,vmax=0.2,vmin=-0.2,
			cmap=plt.cm.jet)

	ax2 = fig.add_subplot(212)
	p2 = ax2.imshow(U[i],vmax=0.2,vmin=-0.2,cmap=plt.cm.jet)
	fig.colorbar(p2, pad=0.11, location = 'bottom', orientation='horizontal')

	return plt

anim = FuncAnimation(fig, animate2, interval=10, frames=Nt, repeat=True)
#anim.save("2d_wave.mp4",writer="ffmpeg")
plt.show()