# Coding Exercise 5 (Python 1) | Jenna Tran

import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3   # second
dt = 1e-3        # second
tau = 20e-3      # second
el = -60e-3      # milivolt
vr = -70e-3      # milivolt
vth = -50e-3     # milivolt
r = 100e6        # ohm
i_mean = 25e-11  # ampere

# Initialize step_end
step_end = 25

with plt.xkcd():
  # Initialize the figure
  plt.figure()
  plt.title('Synaptic Input $I(t)$')
  plt.xlabel('time (s)')
  plt.ylabel('$I$ (A)')

  # Loop for step_end steps
  for step in range(step_end):

    # Compute value of t
    t = step * dt

    # Compute value of i at this time step
    i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

    # Plot i (use 'ko' to get small black dots (short for color='k' and marker = 'o'))
    plt.plot(t, i, 'ko')

  # Display the plot
  plt.show()