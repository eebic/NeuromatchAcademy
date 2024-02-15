# Coding Exercise 2 (Calculus: Numerical Method) | Jenna Tran

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# @title Plotting Functions

time = np.arange(0, 1, 0.01)

## LIF PLOT
def plot_IF(t, V, I, Spike_time):
  """
    Args:
      t  : time
      V  : membrane Voltage
      I  : Input
      Spike_time : Spike_times
    Returns:
      figure with three panels
      top panel: Input as a function of time
      middle panel: membrane potential as a function of time
      bottom panel: Raster plot
  """

  with plt.xkcd():
    fig = plt.figure(figsize=(12,4))
    gs = gridspec.GridSpec(3, 1,  height_ratios=[1, 4, 1])
    # PLOT OF INPUT
    plt.subplot(gs[0])
    plt.ylabel(r'$I_e(nA)$')
    plt.yticks(rotation=45)
    plt.plot(t,I,'g')
    #plt.ylim((2,4))
    plt.xlim((-50,1000))
    # PLOT OF ACTIVITY
    plt.subplot(gs[1])
    plt.plot(t,V,':')
    plt.xlim((-50,1000))
    plt.ylabel(r'$V(t)$(mV)')
    # PLOT OF SPIKES
    plt.subplot(gs[2])
    plt.ylabel(r'Spike')
    plt.yticks([])
    plt.scatter(Spike_time,1*np.ones(len(Spike_time)), color="grey", marker=".")
    plt.xlim((-50,1000))
    plt.xlabel('time(ms)')
    plt.show()

def Euler_Integrate_and_Fire(I, time, dt):
  """
  Args:
    I: Input
    time: time
    dt: time-step
  Returns:
    Spike: Spike count
    Spike_time: Spike times
    V: membrane potential esitmated by the Euler method
  """

  Spike = 0
  tau_m = 10
  R_m = 10
  t_isi = 0
  V_reset = E_L = -75
  n = len(time)
  V = V_reset * np.ones(n)
  V_th = -50
  Spike_time = []

  for k in range(n-1):
    dV = (-(V[k] - E_L) + R_m*I[k]) / tau_m
    V[k+1] = V[k] + dt*dV

    # Discontinuity for Spike
    if V[k] > V_th:
      V[k] = 0
      V[k+1] = V_reset
      t_isi = time[k]
      Spike = Spike + 1
      Spike_time = np.append(Spike_time, time[k])

  return Spike, Spike_time, V

# Set up time step and current
dt = 1
t = np.arange(0, 1000, dt)
I = np.sin(4 * 2 * np.pi * t/1000) + 2

# Model integrate and fire neuron
Spike, Spike_time, V = Euler_Integrate_and_Fire(I, t, dt)

# Visualize
with plt.xkcd():
  plot_IF(t, V,I,Spike_time)

plt.show()