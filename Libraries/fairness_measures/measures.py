# fairness_measures/measures_3.py (or similar utility file)
#
# This file contains functions for calculating benefit and fairness metrics
# based on spatial predictions and real event data across defined regions.

import numpy as np
import pandas as pd

#############################################################################################################################

# Benefit functions
# These functions quantify how well a prediction "benefits" a specific region
# by capturing real events within high-intensity areas.

## Calculate benefit or utility for a group (protected or not) from a prediction.
# This version calculates the mean absolute difference between normalized real events
# and normalized intensity within the masked region. Lower values indicate better alignment.
# Args:
#   mask_protected (np.ndarray): A boolean mask indicating the region of interest (True for cells in the group).
#   real_events (np.ndarray): A 2D NumPy array representing the grid of real event counts or intensity.
#   intensity_matrix (np.ndarray): A 2D NumPy array representing the model's predicted intensity grid.
# Returns:
#   float: The mean absolute difference within the masked region.
# Raises:
#   AssertionError: If input array shapes do not match.
def f_benefit(mask_protected,real_events,intensity_matrix):

    # Assert that input arrays have the same shape
    assert(mask_protected.shape == real_events.shape), f"sizes protected and real events not same"
    assert(mask_protected.shape == intensity_matrix.shape), f"sizes protected and intensity not same"
    assert(real_events.shape == intensity_matrix.shape), f"sizes real events and intensity not same"

    # Normalize real events grid if its sum is not 1 and not 0
    if real_events.sum() != 1:
        if real_events.sum() != 0:
            real_events=real_events/real_events.sum()

    # Normalize intensity matrix if its sum is not 1 and not 0
    if intensity_matrix.sum() != 1:
        if intensity_matrix.sum() != 0:
            intensity_matrix=intensity_matrix/intensity_matrix.sum()

    # Calculate the mean absolute difference between normalized intensity and real events within the masked region
    return abs(intensity_matrix[mask_protected]-real_events[mask_protected]).mean()


## Calculate benefit for the k cells with the most intensity within the masked region.
# This version calculates the mean of the absolute differences for the top k cells
# within the masked region, sorted by predicted intensity.
# Args:
#   mask_protected (np.ndarray): A boolean mask indicating the region of interest.
#   real_events (np.ndarray): Normalized real event grid.
#   intensity_matrix (np.ndarray): Normalized predicted intensity grid.
#   k (int): The number of top cells to consider.
# Returns:
#   float: The mean absolute difference for the top k cells in the masked region.
# Raises:
#   AssertionError: If input array shapes do not match.
def f_benefit_k_most(mask_protected,real_events,intensity_matrix,k):

    # Assert that input arrays have the same shape
    assert(mask_protected.shape == real_events.shape), f"sizes protected and real events not same"
    assert(mask_protected.shape == intensity_matrix.shape), f"sizes protected and intensity not same"
    assert(real_events.shape == intensity_matrix.shape), f"sizes real events and intensity not same"

    # Normalize real events grid if its sum is not 1
    if real_events.sum() != 1:
        real_events=real_events/real_events.sum()

    # Normalize intensity matrix if its sum is not 1
    if intensity_matrix.sum() != 1:
        intensity_matrix=intensity_matrix/intensity_matrix.sum()

    # Create a 2xN array where N is the number of cells in the masked region.
    # Row 0 is intensity within the mask, Row 1 is negative real events within the mask.
    M=np.array([intensity_matrix[mask_protected],-real_events[mask_protected]])
    # Convert to DataFrame, transpose, sort by predicted intensity (column 0) descending,
    # reset index, select the top k rows, and calculate the mean of the absolute sum across rows.
    # The sum(axis=1) calculates intensity - real events for each cell.
    return abs(pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).head(k).sum(axis=1)).mean()

## Calculate benefit for the k-th cell with the most intensity within the masked region.
# This version returns the difference (or its absolute/under/over value)
# for the specific k-th cell when sorted by predicted intensity.
# Args:
#   mask_protected (np.ndarray): A boolean mask indicating the region of interest.
#   real_events (np.ndarray): Normalized real event grid.
#   intensity_matrix (np.ndarray): Normalized predicted intensity grid.
#   k (int): The rank of the cell (0-indexed) when sorted by predicted intensity.
#   abs_ (bool): If True, return the absolute difference.
#   under_over (str or None): If "under", return max(difference, 0). If "over", return max(-difference, 0).
# Returns:
#   float: The calculated difference value for the k-th cell based on abs_ and under_over flags.
# Raises:
#   Exception: If under_over is not None or "under" or "over".
def f_benefit_k_ind(mask_protected,real_events,intensity_matrix,k,abs_=False,under_over=None):

    # Normalize real events grid if its sum is not 1
    if real_events.sum() != 1:
        real_events=real_events/real_events.sum()

    # Normalize intensity matrix if its sum is not 1
    if intensity_matrix.sum() != 1:
        intensity_matrix=intensity_matrix/intensity_matrix.sum()

    # Create a 2xN array (intensity vs negative real events) for cells within the masked region.
    M=np.array([intensity_matrix[mask_protected],-real_events[mask_protected]])

    # Convert to DataFrame, transpose, sort by predicted intensity (column 0) descending,
    # reset index, and select the k-th row (0-indexed).
    # Summing the row gives intensity - real events for the k-th cell.
    R=pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).loc[k].sum()

    # Return the result based on the flags
    if abs_ and under_over==None:
        return abs(R) # Return absolute difference
    elif ~abs_ and under_over==None:
        return R # Return raw difference
    elif under_over=="under":
        return max(R,0) # Return difference if positive, 0 otherwise (under-prediction)
    elif under_over=="over":
        return max(-R,0) # Return negative difference if negative, 0 otherwise (over-prediction)
    else:
        # Raise exception for invalid under_over value
        return Exception()

#############################################################################################################################

# Fairness functions (lower value is better) for the case with just 1 protected variable (two groups)

# AD Absolute difference
# Calculate the absolute difference in benefit between the protected and non-protected regions.
# Args:
#   mask_protected (np.ndarray): A boolean mask for the protected region.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   f_benefit (function): The benefit function to use (default is f_benefit).
#   ben_kwgs (dict): Keyword arguments to pass to the benefit function.
# Returns:
#   float: The absolute difference in benefit between the two groups.
# Raises:
#   Exception: If an error occurs during benefit calculation.
def Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit=f_benefit,ben_kwgs={}):
    try:
        # Calculate benefit for the protected group and the inverse mask (non-protected group), then return the absolute difference.
        return abs(f_benefit(mask_protected,real_events,intensity_matrix,**ben_kwgs)-f_benefit(~mask_protected,real_events,intensity_matrix,**ben_kwgs))
    except Exception as e:
        return e # Return the exception if calculation fails

## Calculate value_unfairness
# This function calculates different average absolute differences in benefit
# for the k-th most predicted cell, averaged over a range of k values.
# Args:
#   mask_protected (np.ndarray): A boolean mask for the protected region.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
# Returns:
#   dict: A dictionary containing average absolute difference ('abs'), average raw difference ('val'),
#         average under-prediction difference ('under'), and average over-prediction difference ('over').
def value_unfairness(mask_protected,real_events,intensity_matrix):
    v_u={} # Initialize dictionary to store results
    # Calculate the minimum size of the protected and non-protected groups
    min_size = min(mask_protected.sum(),(~mask_protected).sum())

    # Calculate average raw difference for k from 0 to min_size-1
    v_u["val"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"abs_":False}) for k in range(0,min_size)]).mean()
    # Calculate average absolute difference for k from 0 to min_size-1
    v_u["abs"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"abs_":True}) for k in range(0,min_size)]).mean()
    # Calculate average under-prediction difference for k from 0 to min_size-1
    v_u["under"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"under_over":"under"}) for k in range(0,min_size)]).mean()
    # Calculate average over-prediction difference for k from 0 to min_size-1
    v_u["over"]=np.array([Abs_diff(mask_protected,real_events,intensity_matrix,f_benefit_k_ind,ben_kwgs={"k":k,"under_over":"over"}) for k in range(0,min_size)]).mean()

    return v_u # Return the dictionary of average differences

#############################################################################################################################

# Fairness functions (lower value is better) for the case with more than 1 protected variable (multiple groups)

## Calculate the square of the difference between the benefit function of two groups.
# Args:
#   mask1 (np.ndarray): Boolean mask for the first group.
#   mask2 (np.ndarray): Boolean mask for the second group.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   ben_kwgs (dict): Keyword arguments to pass to the benefit function.
# Returns:
#   float: The square of the difference in benefit between the two groups.
# Raises:
#   Exception: If an error occurs during benefit calculation.
def square_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        # Calculate benefit for mask1 and mask2, then return the square of their difference.
        return (f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))**2
    except Exception as e:
        return e # Return the exception if calculation fails

## Calculate the absolute difference between the benefit function of two groups.
# Args:
#   mask1 (np.ndarray): Boolean mask for the first group.
#   mask2 (np.ndarray): Boolean mask for the second group.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   ben_kwgs (dict): Keyword arguments to pass to the benefit function.
# Returns:
#   float: The absolute difference in benefit between the two groups.
# Raises:
#   Exception: If an error occurs during benefit calculation.
def abs_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        # Calculate benefit for mask1 and mask2, then return the absolute difference.
        return abs(f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))
    except Exception as e:
        return e # Return the exception if calculation fails

## Calculate the Mean of the square difference between the benefit function of all pairs of groups.
# This is a variance-like measure of fairness across multiple groups.
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
# Returns:
#   float: The mean squared difference in benefit across all pairs of groups.
# Raises:
#   Exception: If an error occurs during calculation.
def variance(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        # Calculate the square difference in benefit for all unique pairs of masks in the dictionary.
        # Then return the mean of these squared differences.
        return np.array([square_diff_benefit(dict_masks[i],dict_masks[j],f_benefit,real_events,intensity_matrix) for i in dict_masks.keys() for j in dict_masks.keys()]).mean()
    except Exception as e:
        return e # Return the exception if calculation fails

## Max difference between benefit function of all pairs of groups (Max-Min difference).
# This is a fairness measure that calculates the difference between the maximum and minimum
# benefit values observed across all defined groups. Lower is better.
# Can also calculate the Ratio (higher is better).
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   measure (str): "Diff" for Max-Min difference (default), "Ratio" for Min/Max ratio.
# Returns:
#   float: The calculated Max-Min difference or Min/Max ratio.
# Raises:
#   Exception: If an error occurs during calculation.
def MM(dict_masks,f_benefit,real_events,intensity_matrix,measure="Diff"):
    try:
        # Calculate the benefit for each mask in the dictionary.
        R=[f_benefit(dict_masks[i],real_events,intensity_matrix) for i in dict_masks]
        if measure == "Diff":
            # Return the difference between the maximum and minimum benefit values.
            return max(R)-min(R)
        if measure == "Ratio":
            # Return the ratio of the minimum to the maximum benefit values (higher is better).
            return min(R)/max(R)

    except Exception as e:
        return e # Return the exception if calculation fails

## Calculate the Gini coefficient.
# This is a measure of statistical dispersion intended to represent the income or wealth
# distribution of a nation's residents, but here applied to the distribution of benefit
# values across groups. Lower is better (more equal distribution of benefit).
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
# Returns:
#   float: The calculated Gini coefficient.
# Raises:
#   Exception: If an error occurs during calculation.
def gini_coeff(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        # Calculate the benefit for each mask and convert to a NumPy array.
        R=np.array([f_benefit(dict_masks[i],real_events,intensity_matrix) for i in dict_masks])
        # R=1-R/R.sum() # Commented out line

        # Calculate the denominator for the Gini coefficient formula.
        den=2*len(R)*sum(R)
        # Calculate the Gini coefficient using the formula:
        # Sum of absolute differences between all pairs of benefit values, divided by (2 * number of groups * sum of benefit values).
        G=abs(R.reshape(len(R),1)-R.reshape(1,len(R))).sum()/den
        return G # Return the calculated Gini coefficient
    except Exception as e:
        return e # Return the exception if calculation fails


## Calculate the Jain's fairness index (higher is better).
# This index measures the "fairness" of a set of values; a value of 1 indicates perfect fairness.
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
# Returns:
#   float: The calculated Jain's fairness index.
# Raises:
#   Exception: If an error occurs during calculation.
def jain_index(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        # Calculate the benefit for each mask and convert to a NumPy array.
        R=np.array([f_benefit(dict_masks[i],real_events,intensity_matrix) for i in dict_masks])
        # R=1-R/R.sum() # Commented out line

        # Calculate the denominator for Jain's index formula.
        den=len(R)*sum(R**2)
        # Calculate Jain's index using the formula:
        # (Sum of benefit values)^2 / (number of groups * sum of squared benefit values).
        J=(R.sum())**2/den # NOTE: Original code had (R.sum())**1, corrected to **2 based on standard formula.
        return J # Return the calculated Jain's index
    except Exception as e:
        return e # Return the exception if calculation fails
