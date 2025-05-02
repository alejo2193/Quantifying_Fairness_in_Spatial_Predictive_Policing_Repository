# poisson/poisson.py (or similar utility file)
#
# This file contains functions for calculating the Poisson likelihood, its gradient,
# and related utilities for parameter optimization.

import numpy as np # For numerical operations
import sparse as sp # For working with sparse arrays
from scipy.special import factorial # For calculating factorials
from scipy.optimize import minimize # For general-purpose optimization
import cvxpy as cp # For defining and solving convex optimization problems (used in likelihood_2d)


#### likelihood (Comment indicating a section for likelihood functions)

# Define a function to compute the Poisson likelihood of the data given parameters and covariates.
# Assumes a log-linear model where log(expected_count) = W @ thetha.
# Likelihood = Product over (time, row, col) of [ exp(-lambda) * lambda^data / data! ]
# Log-Likelihood = Sum over (time, row, col) of [ -lambda + data * log(lambda) - log(data!) ]
# where lambda = exp(W @ thetha).
# Log-Likelihood = Sum over (time, row, col) of [ -exp(W @ thetha) + data * (W @ thetha) - log(data!) ]
# Args:
#   data (sp.COO or np.ndarray): The data matrix (time, row, column) representing event counts.
#   thetha (np.ndarray): A NumPy array of parameters (weights).
#   W (sp.COO or np.ndarray): The covariate matrix.
# Returns:
#   float: The calculated log-likelihood value.
def compute_likelihood(data,thetha,W):
    try:
        # Attempt to calculate log-likelihood assuming sparse data and W.
        # (data * (W @ thetha)) - element-wise product of data and the linear predictor.
        # - np.exp(W @ thetha) - subtract the expected counts (lambda).
        # - np.log(factorial(data.todense())) - subtract log(data!) term. Requires converting sparse data to dense.
        return (data*(W @ thetha)-np.exp(W @ thetha)-np.log(factorial(data.todense()))).sum()
    except:
        # If sparse operations fail or data is dense, perform calculations assuming dense data.
        return (data*(W @ thetha)-np.exp(W @ thetha)-np.log(factorial(data))).sum()


# Define a function to compute the Jacobian (gradient) of the log-likelihood with respect to the parameters (thetha).
# The gradient for the i-th parameter is Sum over (time, row, col) of [ W[:,:,:,i] * (data - exp(W @ thetha)) ].
# The result is then normalized by its L2 norm.
# Args:
#   data (sp.COO or np.ndarray): The data matrix.
#   thetha (np.ndarray): Parameters.
#   W (sp.COO or np.ndarray): The covariate matrix.
# Returns:
#   np.ndarray: The normalized gradient vector.
def jac_like (data,thetha,W):
    # Calculate the gradient components.
    # Iterate through each parameter (i in range(len(thetha))).
    # W[:,:,:,i] selects the covariate vector for the i-th parameter across all time and spatial locations.
    # (data - np.exp(W @ thetha)) is the difference between observed and expected counts.
    # The element-wise product and sum gives the gradient component for parameter i.
    jac_= np.array([(W [:,:,:,i] * (data-np.exp(W @ thetha))).sum() for i in range(len(thetha))])
    # Normalize the gradient vector by its L2 norm.
    return jac_/np.linalg.norm(jac_)

### func to optimize thetha (Comment indicating a section for optimization functions)

# Define an objective function to optimize theta.
# This function calculates the L2 norm of the difference between the sum of data
# across time and the sum of expected counts (exp(W @ thetha)) across time.
# Minimizing this aims to match the total observed counts with the total expected counts per cell.
# Args:
#   data (sp.COO or np.ndarray): The data matrix.
#   thetha (np.ndarray): Parameters.
#   W (sp.COO or np.ndarray): The covariate matrix.
# Returns:
#   float: The L2 norm of the difference between summed observed and expected counts.
def to_opt_theta(data,thetha,W):
    try:
        # Attempt calculation assuming sparse data and W.
        # data.sum(axis=0) sums event counts over time for each spatial cell.
        # np.exp(W @ thetha).sum(axis=0) sums expected counts over time for each spatial cell.
        # .todense() converts sparse results to dense for subtraction and norm calculation.
        return np.linalg.norm( (data.sum(axis=0)- np.exp(W @ thetha).sum(axis=0)).todense())
    except:
        # If sparse operations fail or data is dense, perform calculations assuming dense data.
        return np.linalg.norm( data.sum(axis=0)- np.exp(W @ thetha).sum(axis=0))


# Define a function to compute the best theta parameters by minimizing the 'to_opt_theta' objective function.
# Uses scipy.optimize.minimize with the "Powell" method.
# Args:
#   t_0 (np.ndarray): Initial guess for the parameters (theta).
#   data (sp.COO or np.ndarray): The data matrix.
#   W (sp.COO or np.ndarray): The covariate matrix.
# Returns:
#   np.ndarray: The optimized parameter vector (theta).
def best_theta(t_0,data,W):

    # bnds = [(0, None) for i in range(t_0.size)] # Commented out: Example of bounds definition

    # Perform minimization using scipy.optimize.minimize.
    # The objective function is a lambda that calls to_opt_theta.
    # The initial guess is t_0.
    # The method used is "Powell".
    # Bounds are commented out.
    X=minimize(lambda x : to_opt_theta(data,x,W),t_0,
               method="Powell"
            #    bounds=bnds, # Commented out bounds
               )
    # Print the message from the optimization result (e.g., status of convergence).
    print(X["message"])

    # Return the optimized parameter values.
    return X.x

##### method cvxpy (Comment indicating a section for cvxpy related functions)

# Define a function to calculate the Poisson likelihood for use with cvxpy.
# This version is specifically structured to be compatible with cvxpy's variables and functions.
# Args:
#   data (sp.COO or np.ndarray): The data matrix.
#   thetha (cp.Variable): The cvxpy variable representing the parameters.
#   W (sp.COO or np.ndarray): The covariate matrix.
# Returns:
#   cp.Expression: A cvxpy expression representing the log-likelihood.
def likelihood_2d(data,thetha,W):
    # Calculate the log(data!) term. Requires converting sparse data to dense and reshaping.
    log__= -np.log(factorial(data.reshape((data.shape[0],data.shape[1]*data.shape[2])).todense())).sum()
    # Calculate the linear predictor (W @ thetha). Requires converting W to dense and reshaping.
    linear_predictor = W.todense().reshape((data.size,thetha.size)) @ thetha
    # Calculate the cvxpy expression for the log-likelihood:
    # (data * linear_predictor) - element-wise product of dense data (reshaped) and linear predictor.
    # - cp.sum(cp.exp(linear_predictor)) - subtract the sum of expected counts (lambda).
    # + log__ - add the log(data!) term.
    tempL = data.todense().reshape((data.size,)) @ linear_predictor - cp.sum(cp.exp(linear_predictor)) +log__
    return tempL # Return the cvxpy expression for the log-likelihood
