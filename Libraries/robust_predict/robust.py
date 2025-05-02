# optimization/solver.py (or similar utility file)
#
# This file contains the implementation of an optimization problem solver
# using cvxpy and two optimization algorithms (RSALA and ADGRAD) for finding
# model parameters based on spatial data and likelihood functions.

# Import necessary libraries
import numpy as np # For numerical operations
# Import cvxpy for defining and solving convex optimization problems
import cvxpy as cp
# Import pickle for saving/loading the state of the optimization problem object
import pickle

# Import specific functions from other local utility files.
# Assuming 'poisson' module contains likelihood calculation functions (e.g., compute_likelihood, likelihood_2d, jac_like).
from .poisson import *
# Assuming 'utils' module contains functions like get_best_shift and do_shift.
from .utils import *

# Define a class to represent the convex optimization problem.
# This class formulates a problem likely related to maximizing a lower bound
# on the likelihood, using 'delta' as a variable representing this bound.
class poissonProblemVector:
    # Constructor for the poissonProblemVector class.
    # Initializes the problem with the number of features and an optional solver.
    # Args:
    #   numFeatures (int): The number of features or parameters in the model (dimension of theta).
    #   solver (str, optional): The name of the solver to use (e.g., "SCS", "ECOS"). Defaults to "SCS".
    def __init__(self, numFeatures,solver="SCS"):
        self.numFeatures = numFeatures # Store the number of features
        self.delta = cp.Variable() # Define a cvxpy variable 'delta', likely representing a lower bound or auxiliary variable.
        # Define a cvxpy variable 'theta' for the model parameters, with size numFeatures and no non-negativity constraint.
        self.theta = cp.Variable(numFeatures,nonneg=False)
        self.constraints = [] # Initialize an empty list to store cvxpy constraints.
        self.solver=solver # Store the specified solver name.

        # Define the objective function for the optimization problem.
        # The objective is to maximize 'delta'.
        self.obj = cp.Maximize(self.delta)
        # Create the cvxpy Problem instance with the initial objective and empty constraints.
        self.formulation = cp.Problem(self.obj, self.constraints)

    # Method to add a constraint to the optimization problem.
    # This constraint is based on the likelihood function and the current data/covariates.
    # It enforces that delta is less than or equal to the likelihood for the given data and theta.
    # Args:
    #   data (sp.COO or np.ndarray): The data matrix.
    #   W (sp.COO or np.ndarray): The covariate matrix.
    def addConstraint(self, data,W):
        """
        Adds a constraint based on the current data and covariate matrix.
        The constraint is: delta <= likelihood_2d(data, theta, W)
        """
        # Calculate the likelihood term using the likelihood_2d function (assumed from .poisson).
        # This function likely calculates the likelihood of the data given the parameters theta and covariates W.
        likelihood_term = likelihood_2d(data,self.theta,W)

        # Formulate the constraint: delta <= likelihood_term
        cons = self.delta
        cons -= likelihood_term # Equivalent to self.delta - likelihood_term <= 0

        # Append the formulated constraint to the list of constraints.
        self.constraints.append(cons <= 0)

    # Method to redefine the cvxpy problem with the current set of constraints.
    def redefineProblem(self):
        """
        Updates the problem with the latest set of constraints.
        """
        # Create a new cvxpy Problem instance using the original objective and the current list of constraints.
        self.formulation = cp.Problem(self.obj, self.constraints)

    # Method to solve the current optimization problem.
    # Attempts to use the specified solver, with a fallback to the default solver if an error occurs.
    # Args:
    #   new_solver (str, optional): A different solver to try first. Defaults to None.
    def solve(self,new_solver=None):
        """
        Solve the current optimization problem. If the primary solver returns an error, try without an explicit solver.
        """
        # Print the number of constraints in the current problem formulation.
        print("Attempting to solve problem instance with {} constraints".format(len(self.constraints)))

        try:
            # Attempt to solve the problem using the provided new_solver or the instance's default solver.
            if new_solver:
                self.formulation.solve(solver=new_solver)
            else:
                self.formulation.solve(solver=self.solver)
            # self.formulation.solve(solver='ECOS') # Commented out alternative solver
        except Exception as e: # Catch any exception during solving
            print(f"error in solver: {e}, trying another") # Print the error and indicate trying another solver
            # If the specific solver fails, try solving without specifying a solver (cvxpy will pick one).
            self.formulation.solve()
        # Print the status of the optimization problem after attempting to solve.
        print(self.formulation.status)

    # Method to perform one optimization step: add constraint, redefine, solve, and return theta value.
    # Args:
    #   data (sp.COO or np.ndarray): The data matrix.
    #   W (sp.COO or np.ndarray): The covariate matrix.
    # Returns:
    #   np.ndarray: The value of the theta variable after solving the problem.
    def step_opt(self,data,W):
        # Add a new constraint based on the current data and covariates.
        self.addConstraint(data,W)
        # Redefine the problem formulation with the new constraint.
        self.redefineProblem()
        # Solve the updated optimization problem.
        self.solve()
        # Return the value of the theta variable as a NumPy array.
        return np.array(list(self.theta.value))

    # Method to save the current state of the poissonProblemVector object to a file using pickle.
    # Args:
    #   path_file (str): The path to the file where the state will be saved.
    def save_state(self,path_file):
        # Open the specified file in binary write mode.
        with open(path_file, 'wb') as f:
            # Use pickle.dump to serialize and save the current object instance to the file.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


# Implement the RSALA (Regularized Spatial Adaptive Learning Algorithm) optimization algorithm.
# This algorithm iteratively updates the model parameters (theta) by adding constraints
# based on data that has been "shifted" according to the current parameters.
# Args:
#   data (sp.COO or np.ndarray): The initial data matrix.
#   W (sp.COO or np.ndarray): The covariate matrix.
#   neighborhood (dict): Dictionary mapping cell positions to their neighbors.
#   likelihood_func (function, optional): The likelihood function to use for calculating best shift. Defaults to compute_likelihood.
#   epsilon (float, optional): Convergence tolerance. Defaults to 1e-5.
#   solver (str, optional): Solver to use for the optimization problem. Defaults to "SCS".
#   max_iter (int, optional): Maximum number of iterations. Defaults to 20.
#   save (str, optional): Path to save the optimization problem state. Defaults to None.
# Returns:
#   tuple: A tuple containing:
#          - thetha_new (np.ndarray): The final optimized parameters.
#          - update_data (sp.COO or np.ndarray): The data matrix after the last shift.
def RSALA(data,W,neighborhood,likelihood_func=compute_likelihood,epsilon=1e-5,solver="SCS",max_iter=20,save=None):
    # Initialize the optimization problem solver object.
    optProb = poissonProblemVector(W.shape[-1],solver=solver)
    # Add the initial constraint based on the original data.
    optProb.addConstraint(data,W)
    # Redefine the problem formulation.
    optProb.redefineProblem()
    # Solve the initial problem to get the first estimate of theta.
    optProb.solve()

    # Get the initial parameter values.
    thetha_new = np.array(list(optProb.theta.value))

    gap=10 # Initialize gap with a value larger than epsilon to enter the loop.
    iter_ = 0 # Initialize iteration counter.

    # Start the main optimization loop. Continue as long as the absolute gap is greater than epsilon
    # and the maximum number of iterations has not been reached.
    while abs(gap)> epsilon and iter_ <max_iter:
        iter_+=1 # Increment iteration counter.
        thetha_current=thetha_new.copy() # Store the current parameter values.

        # Get the best shift coding based on the current parameters and data.
        # Uses the get_best_shift function (assumed from .utils).
        best_shift=get_best_shift(thetha_current,data,neighborhood,W,likelihood_func)
        # Apply the best shift to the data to get the updated data matrix.
        # Uses the do_shift function (assumed from .utils).
        update_data=do_shift(best_shift,data,neighborhood)

        try:
            # Perform one step of the optimization problem with the updated data.
            # This adds a new constraint and solves for a new theta.
            thetha_new=optProb.step_opt(update_data,W)

            # Calculate the gap: difference in likelihood between the updated data with new theta
            # and the updated data with the current theta. This measures progress.
            # Uses the compute_likelihood function (assumed from .poisson).
            gap = compute_likelihood(update_data,thetha_new,W) - compute_likelihood(update_data,thetha_current,W)

            # If a save path is provided, save the state of the optimization problem object.
            if save:
                optProb.save_state(save)
        except Exception as e: # Catch any exception during the optimization step
            print(f"Finish model in iteration {iter_} due to error: {e}") # Print error and iteration number
            break # Exit the loop if an error occurs

    # Return the final parameter values and the data matrix after the last shift.
    return thetha_new, update_data


# Implement the ADGRAD (Adaptive Gradient) optimization algorithm.
# This algorithm iteratively updates model parameters using gradient ascent
# on the likelihood function, with an adaptive step size determined by backtracking line search.
# It also incorporates the spatial shifting concept.
# Args:
#   data (sp.COO or np.ndarray): The initial data matrix.
#   W (sp.COO or np.ndarray): The covariate matrix.
#   neighborhood (dict): Dictionary mapping cell positions to their neighbors.
#   likelihood_func (function, optional): The likelihood function to use for calculating best shift. Defaults to compute_likelihood.
#   epsilon (float, optional): Convergence tolerance. Defaults to 1e-5.
#   solver (str, optional): Solver to use for the initial optimization problem. Defaults to "SCS".
#   max_iter (int, optional): Maximum number of iterations. Defaults to 50.
#   save (str, optional): Path to save the optimization problem state (only saves the initial state in this implementation). Defaults to None.
# Returns:
#   tuple: A tuple containing:
#          - thetha_new (np.ndarray): The final optimized parameters.
#          - update_data (sp.COO or np.ndarray): The data matrix after the last shift.
def ADGRAD (data,W,neighborhood,likelihood_func=compute_likelihood,epsilon=1e-5,solver="SCS",max_iter=50,save=None):

    # Initialize the optimization problem solver object (used only for the initial theta estimate).
    optProb = poissonProblemVector(W.shape[-1],solver=solver)
    # Add the initial constraint based on the original data.
    optProb.addConstraint(data,W)
    # Redefine the problem formulation.
    optProb.redefineProblem()
    # Solve the initial problem to get the first estimate of theta.
    optProb.solve()

    # Get the initial parameter values.
    thetha_new = np.array(list(optProb.theta.value))

    gap=10 # Initialize gap with a value larger than epsilon to enter the loop.
    iter_ = 0 # Initialize iteration counter.

    # Perform initial backtracking line search to find a suitable step size (alpha).
    # Minimizing -likelihood is equivalent to maximizing likelihood, so we use -jac_like for gradient.
    alpha= BacktrackingLineSearch(100,thetha_new, # Initial alpha guess
                                  -jac_like(data,thetha_new,W), # Search direction (negative gradient)
                                  lambda x: -compute_likelihood(data,x,W), # Objective function (negative likelihood)
                                  lambda x: -jac_like(data,x,W), # Gradient of the objective function
                                  rho=0.8,c=0.3) # Line search parameters

    # Start the main optimization loop. Continue as long as the absolute gap is greater than epsilon
    # and the maximum number of iterations has not been reached.
    while abs(gap)> epsilon and iter_ <=max_iter:
        iter_+=1 # Increment iteration counter.

        thetha_current=thetha_new.copy() # Store the current parameter values.

        # Get the best shift coding based on the current parameters and data.
        best_shift=get_best_shift(thetha_current,data,neighborhood,W,likelihood_func)
        # Apply the best shift to the data to get the updated data matrix.
        update_data=do_shift(best_shift,data,neighborhood)

        # Update the parameters using gradient ascent with the calculated step size (alpha).
        # Uses the Jacobian (gradient) of the likelihood function (jac_like, assumed from .poisson).
        thetha_new= thetha_current + alpha * jac_like(update_data,thetha_current,W)

        # Perform backtracking line search again to find a new step size for the next iteration.
        # Uses the updated data for likelihood and gradient calculations.
        alpha= BacktrackingLineSearch(alpha,thetha_new, # Current alpha and new theta
                                  -jac_like(update_data,thetha_new,W), # Search direction (negative gradient on updated data)
                                  lambda x: -compute_likelihood(update_data,x,W), # Objective function (negative likelihood on updated data)
                                  lambda x: -jac_like(update_data,x,W), # Gradient of objective function on updated data
                                  rho=0.8,c=0.3) # Line search parameters

        # Calculate the gap: difference in likelihood between the updated data with new theta
        # and the updated data with the current theta.
        gap = compute_likelihood(update_data,thetha_new,W) - compute_likelihood(update_data,thetha_current,W)
        # Print iteration progress and gap/alpha values.
        print("Iteration: ",iter_," gap: ", gap, "alpha: ", alpha)

        # If a save path is provided, save the state of the optimization problem object (only the initial state is saved here).
        # NOTE: This save is inside the loop, but the optProb object is not updated within the loop in ADGRAD.
        # Saving inside the loop might be intended to save the *theta_new* values or the *update_data*,
        # but the current implementation saves the initial optProb state repeatedly.
        if save:
            optProb.save_state(save)

    # Return the final parameter values and the data matrix after the last shift.
    return thetha_new, update_data
