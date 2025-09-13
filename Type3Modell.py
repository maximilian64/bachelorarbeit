import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

K = 6
r=1
c=2
d=2
e=1
mu=0.4
IC= (1.8,2.2)
IC2=(7,0.2)

def fp (K,r,c,d,e,mu):  # returns fixed point for parameters
    N=d*math. sqrt((mu/(e-mu)))
    P=(1-N/K)*(d*d+N*N)*(r/(c*N))
    return(N,P)

def RMA(t, z):  # model in format for ivp solve
    N, P = z
    dNdt = r*N * (1 - N/K) - (c * P*N*N) / (d*d + N*N)
    dPdt = P * ((e * N*N) / (d*d + N*N) - mu)
    return [dNdt, dPdt]
def RMA_change(N, P): # model, use this for phase portrait
    dNdt = r*N * (1 - N/K) - (c * P*N*N) / (d*d + N*N)
    dPdt = P * ((e * N*N) / (d*d + N*N) - mu)
    return dNdt, dPdt

# Create grid and points to evaluate functions at
N_vals = np.linspace(0, 10, 20)
P_vals = np.linspace(0, 10, 20)
N_grid, P_grid = np.meshgrid(N_vals, P_vals)
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

U, V = RMA_change(N_grid, P_grid)  # save rates of changes at grid points for phase portraits
sol = solve_ivp(RMA, t_span, IC, t_eval=t_eval) # calculate trajectory
N_traj, P_traj = sol.y
sol2 = solve_ivp(RMA, t_span, IC2, t_eval=t_eval) # calculate another trajectory
N_traj2, P_traj2 = sol2.y

#N_line = np.linspace(0.01, 7, 500)   # isoclines
#P_isocline = (1 - N_line/K) * (1 + N_line) / m
#N_iso_pred = mu / (m - mu)

plt.figure(figsize=(6,6))
plt.streamplot(N_grid, P_grid, U, V, color='blue', cmap='viridis',density=2, linewidth=1, arrowsize=1)#phase portrait
plt.plot(N_traj, P_traj, color="red", lw=2,label=f"Trajektorie mit Startpunkt {IC}") # plot trajectories
plt.plot(N_traj2, P_traj2, color="black", lw=2, label=f"Trajektorie mit Startpunkt {IC2}")
plt.scatter(fp(K,r,c,d,e,mu)[0], fp (K,r,c,d,e,mu)[1], color='y', s=40, label='Innere Ruhelage') # draw fixedpoint
plt.scatter(K, 0, color='orange', s=40, label='(K,0)') #draw(K,0)
#plt.plot(N_line, P_isocline, 'r', lw=2, label='dN/dt=0')
#plt.axvline(x=N_iso_pred, color='g', lw=2, label='dP/dt=0')
#plt.axvline(x=(K-1)/2,color='b',linestyle='--',label='(K-1)/2')

plt.xlabel("Beutetiere")
plt.ylabel("Raubtiere")
plt.title("Phasenportrait und Trajektorien Type 3 Modell")
plt.legend()
plt.grid(alpha=0.3)  
plt.xlim(0,7) 
plt.ylim(0,3)
plt.show()
