# Coding Exercise 1.1 (Statistics) | Jenna Tran

# Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm  # the normal probability distribution

# @title Plotting Functions

def plot_random_sample(x, y, figtitle = None):
  """ Plot the random sample between 0 and 1 for both the x and y axes.

    Args:
      x (ndarray): array of x coordinate values across the random sample
      y (ndarray): array of y coordinate values across the random sample
      figtitle (str): title of histogram plot (default is no title)

    Returns:
      Nothing.
  """
  fig, ax = plt.subplots()
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  plt.xlim([-0.25, 1.25]) # set x and y axis range to be a bit less than 0 and greater than 1
  plt.ylim([-0.25, 1.25])
  plt.scatter(dataX, dataY)
  if figtitle is not None:
    fig.suptitle(figtitle, size=16)
  plt.show()

def generate_random_sample(num_points):
  """ Generate a random sample containing a desired number of points (num_points)
  in the range [0, 1] using a random number generator object.

  Args:
    num_points (int): number of points desired in random sample

  Returns:
    dataX, dataY (ndarray, ndarray): arrays of size (num_points,) containing x
    and y coordinates of sampled points

  """

  # Generate desired number of points uniformly between 0 and 1 (using uniform) for
  #     both x and y
  dataX = np.random.uniform(0, 1, size = (num_points,))
  dataY = np.random.uniform(0, 1, size = (num_points,))

  return dataX, dataY

# Set a seed
np.random.seed(0)

# Set number of points to draw
num_points = 10

# Draw random points
dataX, dataY = generate_random_sample(num_points)

# Visualize
with plt.xkcd():
  plot_random_sample(dataX, dataY, "Random sample of 10 points")

plt.show()