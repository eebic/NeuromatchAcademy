# Coding Exercise 2.2 (Statistics 2) | Jenna Tran

# Imports
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.stats import norm  # the normal probability distribution


# We define the function to optimise, the negative log likelihood
def negLogLike(theta, x):
  """ Function for computing the negative log-likelihood given the observed data
      and given parameter values stored in theta.

      Args:
        theta (ndarray): normal distribution parameters
                        (mean is theta[0], standard deviation is theta[1])
        x (ndarray): array with observed data points

      Returns:
        Calculated negative Log Likelihood value!
  """
  return -sum(np.log(norm.pdf(x, theta[0], theta[1])))

# Set random seed
np.random.seed(0)

# Generate data
true_mean = 5
true_standard_dev = 1
n_samples = 1000
x = np.random.normal(true_mean, true_standard_dev, size=(n_samples, ))

# Define bounds, var has to be positive
bnds = ((None, None), (0, None))

# Optimize with scipy!
optimal_parameters = sp.optimize.minimize(negLogLike, (2, 2), args=x, bounds=bnds)
print(f"The optimal mean estimate is: {optimal_parameters.x[0]}")
print(f"The optimal standard deviation estimate is: {optimal_parameters.x[1]}")

# optimal_parameters contains a lot of information about the optimization,
# but we mostly want the mean and standard deviation