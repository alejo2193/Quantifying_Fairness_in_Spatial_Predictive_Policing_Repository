# evaluation/metrics.py (or similar utility file)
#
# This file contains functions for calculating various evaluation metrics
# used in crime prediction and spatial analysis, such as PAI, PEI, and Weighted Travel.

import numpy as np # For numerical operations
import sparse as sp # For working with sparse arrays (imported, but not directly used in these functions)

from scipy.linalg import norm # For calculating vector norms

# Define a function to calculate break points for binning data.
# Args:
#   min (float): The minimum value of the data range.
#   max (float): The maximum value of the data range.
#   bins (int, optional): The number of bins to create. Defaults to 3.
# Returns:
#   np.ndarray: A NumPy array of bin edges (break points).
def get_break_points(min,max,bins=3):
    # np.linspace creates a sequence of evenly spaced values within a specified interval.
    # bins+1 points are needed to define 'bins' intervals.
    return np.linspace(min,max,bins+1)


# Define a function to generate boolean masks for data based on break points (bins).
# Args:
#   data (np.ndarray): The input NumPy array.
#   break_points (np.ndarray): A NumPy array of bin edges (break points).
# Returns:
#   list: A list of boolean NumPy arrays, where each array is True for data points falling into a specific bin.
def mask_gen(data,break_points):
    mask_sqc=[] # Initialize an empty list to store the masks

    # Iterate through the break points to define the bins.
    # The loop goes up to the second-to-last break point to define intervals.
    for i in range(len(break_points)-1):

        if i!=len(break_points)-2:
            # For all bins except the last one, the interval is [inclusive, exclusive).
            mask = (data >= break_points [i]) & (data < break_points [i+1])

        else:
            # For the last bin, the interval is [inclusive, inclusive].
            mask = (data >= break_points [i]) & (data <= break_points [i+1])

        mask_sqc.append(mask) # Append the generated mask to the list

    return mask_sqc # Return the list of masks


# Define a function to calculate the mean rate within each bin defined by masks.
# Args:
#   data (np.ndarray): The input NumPy array of values.
#   mask_sqc (list): A list of boolean masks, where each mask defines a bin.
# Returns:
#   np.ndarray: A NumPy array of the mean values within each bin. NaN values are replaced with 0.
def rates_bins(data,mask_sqc):
    # Calculate the mean of the data for each mask (bin).
    # Use a list comprehension to iterate through the masks and calculate the mean for the masked data.
    # np.nan_to_num replaces any NaN values (which can occur if a bin is empty) with 0.
    return np.nan_to_num(np.array([data[mask].mean() for mask in mask_sqc]))


# Define a function to calculate the norm of the difference between two vectors.
# Args:
#   v1 (np.ndarray): The first vector.
#   v2 (np.ndarray): The second vector.
#   ord (int or float, optional): The order of the norm (e.g., 1 for L1, 2 for L2, np.inf for maximum absolute difference). Defaults to np.inf.
# Returns:
#   float: The calculated norm of the difference.
def norm_rates(v1,v2,ord=np.inf):
    # Calculate the difference between the two vectors and then compute the specified norm.
    return norm(v1-v2,ord)


# Define a function to compare the distribution of real and predicted rates across bins.
# Calculates the norm of the difference between the mean rates in corresponding bins.
# Args:
#   real (np.ndarray): The array of real values.
#   pred (np.ndarray): The array of predicted values.
#   min (float, optional): Minimum value for binning. Defaults to None (uses real.min()).
#   max (float, optional): Maximum value for binning. Defaults to None (uses real.max()).
#   bins (int, optional): Number of bins. Defaults to 3.
#   ord (int or float, optional): Order of the norm. Defaults to np.inf.
# Returns:
#   float: The norm of the difference between the binned rates of real and predicted values.
def compare_rates_bins(real,pred,min=None,max=None,bins=3,ord=np.inf):
    # Determine the minimum value for binning (use real.min() if not provided).
    if not min:
        min= real.min()

    # Determine the maximum value for binning (use real.max() if not provided).
    if not max:
        max= real.max()

    # Get the break points for the bins based on the min/max range and number of bins.
    break_points=get_break_points(min,max,bins=bins)
    # Generate boolean masks for the real data based on the break points.
    masks=mask_gen(real,break_points)

    # Calculate the mean real rates within each bin.
    rates_real=rates_bins(real,masks)
    # Calculate the mean predicted rates within the same bins (using the masks generated from real data).
    rates_pred=rates_bins(pred,masks)

    # Calculate and return the norm of the difference between the binned real and predicted rates.
    return norm_rates(rates_real,rates_pred,ord)


# Define a function to normalize a sparse 3D data matrix to rates (proportions) per time step.
# Divides each cell's value by the sum of values in that time step.
# Args:
#   data_sparse (sp.COO): The input sparse 3D data matrix (time, row, column).
# Returns:
#   np.ndarray: A dense NumPy array of normalized rates per time step.
def normalize_to_rates(data_sparse):
    # Calculate the sum of values across the spatial dimensions (row and column) for each time step.
    # keepdims=True ensures the resulting sum has the same number of dimensions as the input.
    # Divide the sparse data by these sums (broadcasting will apply the sum to each spatial cell in a time step).
    # Convert the result to a dense NumPy array.
    return (data_sparse/data_sparse.sum(axis=[1,2],keepdims=True)).todense()


#######################  PAI & PEI (Comment indicating a section for PAI and PEI metrics)

# Define a function to rank cells based on their rates and select the top cells that sum up to a certain alpha proportion.
# Args:
#   alpha (float): The target cumulative proportion (between 0 and 1).
#   rates_map (np.ndarray): A 2D NumPy array representing rates or intensity on a spatial grid.
# Returns:
#   np.ndarray: A 1D NumPy array of the flattened indices of the selected top cells.
def rank_cells(alpha,rates_map):
    # Flatten the rates map and get the indices that would sort the flattened array in descending order.
    indices = np.argsort(rates_map.flatten())[::-1]
    n_cells=1 # Initialize cell counter
    # Iterate, adding cells from the sorted list, until the cumulative sum of rates reaches or exceeds alpha.
    while sum(rates_map.flatten()[indices[:n_cells]])< alpha:
        n_cells+=1
    # If the cumulative sum exceeds alpha and more than one cell was included,
    # decrement n_cells to find the minimum number of cells needed to reach exactly or just under alpha.
    if sum(rates_map.flatten()[indices[:n_cells]]) > alpha and n_cells>1:
        n_cells-=1

    # Return the flattened indices of the selected top n_cells.
    cells_index=indices[:n_cells]

    return cells_index # Return the indices of the top cells


# Define a function to calculate PAI (Predictive Accuracy Index) for a single time step.
# PAI = (Proportion of real events in top predicted cells) / (Proportion of area covered by top predicted cells)
# Args:
#   alpha (float): The target cumulative proportion for selecting top predicted cells.
#   real (np.ndarray): A 2D NumPy array of real event counts for the time step.
#   hot_pred (np.ndarray): A 2D NumPy array of predicted intensity/rates for the time step.
# Returns:
#   float: The PAI value for the single time step.
def PAI_one_time(alpha,real,hot_pred):

    # Get the flattened indices of the top predicted cells based on the alpha proportion.
    cells_index=rank_cells(alpha,hot_pred)

    # Calculate 'n': the sum of real events in the selected top predicted cells.
    n=real.flatten()[cells_index].sum()
    # Calculate 'N': the total number of real events in the entire grid.
    N=real.sum()
    # Calculate the proportion of real events captured by the top predicted cells (n/N).
    # Calculate the proportion of the total area covered by the top predicted cells (len(cells_index)/real.size).
    # Return the ratio of these two proportions (PAI).
    return (n/N)/(len(cells_index)/real.size)

# Define a function to calculate the average PAI over multiple time steps.
# Args:
#   alpha (float): The target cumulative proportion for selecting top predicted cells.
#   real (np.ndarray): A 3D NumPy array of real event counts (time, row, col).
#   hot_pred (np.ndarray): A 3D NumPy array of predicted intensity/rates (time, row, col).
# Returns:
#   float: The average PAI value across all time steps.
def PAI(alpha,real,hot_pred):

    values=[] # Initialize list to store PAI for each time step
    # Iterate through each time step.
    for i in range(len(real)):
        # Calculate PAI for the current time step and append to the list.
        values.append(PAI_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values) # Return the mean of PAI values across time steps


# Define a function to calculate PEI (Predictive Efficiency Index) for a single time step.
# PEI = (Density of real events in top predicted cells) / (Density of real events in top real cells)
# Density = number of events / number of cells
# Args:
#   alpha (float): The target cumulative proportion for selecting top cells.
#   real (np.ndarray): A 2D NumPy array of real event counts for the time step.
#   hot_pred (np.ndarray): A 2D NumPy array of predicted intensity/rates for the time step.
# Returns:
#   float: The PEI value for the single time step.
def PEI_one_time(alpha,real,hot_pred):

    # Get the flattened indices of the top predicted cells based on the alpha proportion.
    cells_index_pred=rank_cells(alpha,hot_pred)
    # Calculate 'n': the sum of real events in the selected top predicted cells.
    n=real.flatten()[cells_index_pred].sum()

    # Get the flattened indices of the top *real* cells based on the alpha proportion.
    cells_index_real=rank_cells(alpha,real)
    # Calculate 'n_ast': the sum of real events in the selected top *real* cells.
    n_ast=real.flatten()[cells_index_real].sum()

    # Calculate the density of real events in top predicted cells (n / number of top predicted cells).
    # Calculate the density of real events in top real cells (n_ast / number of top real cells).
    # Return the ratio of these two densities (PEI).
    return (n/len(cells_index_pred))/(n_ast/len(cells_index_real))

# Define a function to calculate the average PEI over multiple time steps.
# Args:
#   alpha (float): The target cumulative proportion for selecting top cells.
#   real (np.ndarray): A 3D NumPy array of real event counts (time, row, col).
#   hot_pred (np.ndarray): A 3D NumPy array of predicted intensity/rates (time, row, col).
# Returns:
#   float: The average PEI value across all time steps.
def PEI(alpha,real,hot_pred):

    values=[] # Initialize list to store PEI for each time step
    # Iterate through each time step.
    for i in range(len(real)):
        # Calculate PEI for the current time step and append to the list.
        values.append(PEI_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values) # Return the mean of PEI values across time steps


# Define a function to calculate PEI* (PEI star) for a single time step.
# PEI* = (Proportion of real events in top predicted cells) / (Proportion of real events in the top k *real* cells, where k is the number of top predicted cells)
# Args:
#   alpha (float): The target cumulative proportion for selecting top predicted cells.
#   real (np.ndarray): A 2D NumPy array of real event counts for the time step.
#   hot_pred (np.ndarray): A 2D NumPy array of predicted intensity/rates for the time step.
# Returns:
#   float: The PEI* value for the single time step.
def PEI_ast_one_time(alpha,real,hot_pred):

    # Get the flattened indices of the top predicted cells based on the alpha proportion.
    cells_index_pred=rank_cells(alpha,hot_pred)
    # Calculate 'n': the sum of real events in the selected top predicted cells.
    n=real.flatten()[cells_index_pred].sum()

    # Get the sum of real events in the top k *real* cells, where k is the number of top predicted cells.
    # Sort flattened real events in descending order and sum the top k values.
    n_ast=np.sort(real,axis=None)[::-1][:len(cells_index_pred)].sum()

    # Return the ratio of the sum of real events in top predicted cells to the sum of real events in the top k real cells.
    return n/n_ast


# Define a function to calculate the average PEI* over multiple time steps.
# Args:
#   alpha (float): The target cumulative proportion for selecting top predicted cells.
#   real (np.ndarray): A 3D NumPy array of real event counts (time, row, col).
#   hot_pred (np.ndarray): A 3D NumPy array of predicted intensity/rates (time, row, col).
# Returns:
#   float: The average PEI* value across all time steps.
def PEI_ast(alpha,real,hot_pred):

    values=[] # Initialize list to store PEI* for each time step
    # Iterate through each time step.
    for i in range(len(real)):
        # Calculate PEI* for the current time step and append to the list.
        values.append(PEI_ast_one_time(alpha,real[i],hot_pred[i]))

    return np.mean(values) # Return the mean of PEI* values across time steps


######################### weighted travel (Comment indicating a section for Weighted Travel metric)

# Define a function to calculate Weighted Travel for a single time step.
# This function appears to be incomplete or uses variables ('res_opt') not defined within its scope.
# It seems intended to calculate a weighted sum based on optimization results and intensity.
# Args:
#   res_opt: Optimization results (structure is unclear).
#   M_intensity (np.ndarray): A 2D NumPy array representing intensity on a spatial grid for a single time step.
# Returns:
#   float: The calculated Weighted Travel for the single time step.
def get_Weighted_travel(res_opt,M_intensity):
    # This calculation iterates through 'res_opt', extracts coordinates and a value,
    # and multiplies the value by the intensity at those coordinates, then sums.
    # The structure of 'res_opt' (list of tuples/lists with string coordinates) is unusual.
    return sum([M_intensity[int(i[1].split(",")[0]),
                int(i[1].split(",")[1])
                ]*i[2] for i in res_opt])

# Define a function to calculate the average Weighted Travel over multiple time steps.
# This function appears to be incomplete or uses variables ('res_opt') not defined within its scope.
# Args:
#   res_opt: Optimization results (structure is unclear).
#   intensity (np.ndarray): A 3D NumPy array representing intensity over time and space.
# Returns:
#   np.ndarray: A NumPy array of Weighted Travel values for each time step.
def Weighted_travel(res_opt,intensity):

    values=[] # Initialize list to store Weighted Travel for each time step
    # Iterate through each time step.
    for i in range(len(intensity)):
        # Calculate Weighted Travel for the current time step using get_Weighted_travel
        # and append to the list.
        # NOTE: 'res_opt' is used here but its origin is unclear.
        values.append(get_Weighted_travel(res_opt,intensity[i]))

    return np.array(values) # Return the array of Weighted Travel values
