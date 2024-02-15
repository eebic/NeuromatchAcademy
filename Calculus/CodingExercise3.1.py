# Coding Exercise 3.1 (Calculus: Numerical Method) | Jenna Tran

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# @title Plotting Functions

time = np.arange(0, 1, 0.01)

def plot_rErI(t, r_E, r_I):
  """
    Args:
      t   : time
      r_E : excitation rate
      r_I : inhibition rate

    Returns:
      figure of r_I and r_E as a function of time

  """
  with plt.xkcd():
    fig = plt.figure(figsize=(6,4))
    plt.plot(t,r_E,':',color='b',label=r'$r_E$')
    plt.plot(t,r_I,':',color='r',label=r'$r_I$')
    plt.xlabel('time(ms)')
    plt.legend()
    plt.ylabel('Firing Rate (Hz)')
    plt.show()

def Euler_Simple_Linear_System(t, dt):
  """
  Args:
    time: time
  dt  : time-step
  Returns:
    r_E : Excitation Firing Rate
    r_I : Inhibition Firing Rate

  """

  # Set up parameters
  tau_E = 100
  tau_I = 120
  n = len(t)
  r_I = np.zeros(n)
  r_I[0] = 20
  r_E = np.zeros(n)
  r_E[0] = 30

  # Loop over time steps
  for k in range(n-1):

    # Estimate r_e
    dr_E = -r_I[k]/tau_E
    r_E[k+1] = r_E[k] + dt*dr_E

    # Estimate r_i
    dr_I = r_E[k]/tau_I
    r_I[k+1] = r_I[k] + dt*dr_I

  return r_E, r_I

# Set up dt, t
dt = 0.1 # time-step
t = np.arange(0, 1000, dt)

# Run Euler method
r_E, r_I = Euler_Simple_Linear_System(t, dt)

# Visualize
with plt.xkcd():
  plot_rErI(t, r_E, r_I)

plt.show()