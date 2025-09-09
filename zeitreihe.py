import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""K = 6
m = 1
mu = 2/3
"""
K = 6
m = 1
mu = 2/3
IC= (7,0.5)

def RMA(t, z):  # model in format for ivp solve
    N, P = z
    dNdt = N * (1 - N/K - (m * P) / (1 + N))
    dPdt = P * ((m * N) / (1 + N) - mu)
    return [dNdt, dPdt]

# Create grid and points to evaluate functions at
N_vals = np.linspace(0, 10, 20)
P_vals = np.linspace(0, 10, 20)
N_grid, P_grid = np.meshgrid(N_vals, P_vals)
t_span = (0, 100)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

sol = solve_ivp(RMA, t_span, IC, t_eval=t_eval) # calculate trajectory
N_traj, P_traj = sol.y
plt.figure(figsize=(10, 5))

plt.plot(t_eval, N_traj, label="Beutetierpopulation", color="blue")
plt.plot(t_eval, P_traj, label="Raubtierpopulation", color="orange")

plt.xlabel("Zeit")
plt.ylabel("Größe der Populationen")
plt.title("Zeitreihe RMA")
plt.legend()
plt.grid(True)
plt.show()
