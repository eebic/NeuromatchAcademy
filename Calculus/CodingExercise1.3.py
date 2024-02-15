# Coding Exercise 1.3 (Calculus: Numerical Method) | Jenna Tran

# Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

# @title Plotting Functions

time = np.arange(0, 1, 0.01)

def visualize_population_approx(t, p):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(t, np.exp(0.3*t), 'k', label='Exact Solution')

    plt.plot(t, p,':o', label='Euler Estimate')
    plt.vlines(t, p, np.exp(0.3*t),
              colors='r', linestyles='dashed', label=r'Error $e_k$')

    plt.ylabel('Population (millions)')
    plt.legend()
    plt.xlabel('Time (years)')
    plt.show()

# Time step
dt = 1

# Make time range from 1 to 5 years with step size dt
t = np.arange(1, 5+dt/2, dt)

# Get number of steps
n = len(t)

# Initialize p array
p = np.zeros(n)
p[0] = np.exp(0.3*t[0]) # initial condition

# Loop over steps
for k in range(n-1):

  # Calculate the population step
    p[k+1] = p[k] + dt * 0.3 * p[k]

# Visualize
with plt.xkcd():
  visualize_population_approx(t, p)

plt.show()