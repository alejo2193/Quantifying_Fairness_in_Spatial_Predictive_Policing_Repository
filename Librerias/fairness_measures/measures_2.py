# fairness_measures/measures_X.py (or similar utility file)
#
# This file contains functions for calculating benefit and fairness metrics,
# with a focus on evaluating performance (confusion matrix, precision, accuracy)
# within the 'k' most predicted cells of specific regions.

import numpy as np
import pandas as pd
from sklearn import metrics # Imported, but not used in the functions provided here.

#############################################################################################################################

# Benefit functions
# These functions quantify how well a prediction "benefits" a specific region
# by capturing real events within high-intensity areas, or evaluate performance
# within the top predicted cells.

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

    # Normalize intensity matrix if its sum != 1 and != 0
    if intensity_matrix.sum() != 1:
        if intensity_matrix.sum() != 0:
            intensity_matrix=intensity_matrix/intensity_matrix.sum()

    # Calculate the mean absolute difference between normalized intensity and real events within the masked region
    return abs(intensity_matrix[mask_protected]-real_events[mask_protected]).mean()


## Calculate benefit/performance for the k cells with the most intensity within the masked region.
# This version calculates and returns confusion matrix components, precision, and accuracy
# based on whether real events occurred in the top k predicted cells within the masked region.
# Args:
#   mask_protected (np.ndarray): A boolean mask indicating the region of interest.
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   k (int): The number of top cells to consider.
# Returns:
#   tuple: A tuple containing:
#          - confusion_matrix_m (np.ndarray): A 2x2 confusion matrix ([TP, FP], [FN, TN]).
#          - precision (float or None): Precision score (TP / (TP + FP)), None if denominator is zero.
#          - accuracy (float or None): Accuracy score ((TP + TN) / Total), None if total is zero.
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

    # Calculate the number of cells with real events within the masked region.
    top_k_real_events = sum(real_events[mask_protected]!=0)
    # If k is greater than the number of cells with real events, cap k at that number.
    # NOTE: This logic seems unusual for selecting top k *predicted* cells.
    # It might be intended to ensure k doesn't exceed the number of relevant cells.
    if k <= top_k_real_events:
        pass # k is valid
    else:
        k = top_k_real_events # Cap k

    # Create a 2xN array where N is the number of cells in the masked region.
    # Row 0 is intensity within the mask, Row 1 is real events within the mask.
    M=np.array([intensity_matrix[mask_protected],real_events[mask_protected]])

    # Convert to DataFrame, transpose, sort by predicted intensity (column 0) descending,
    # reset index, and select the top k rows.
    # Convert values to boolean: True if > 0, False otherwise.
    pandas_top=pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).head(k) > 0

    # Calculate confusion matrix components based on the boolean values in the top k cells.
    # Column 0 represents predicted intensity > 0 (Positive prediction).
    # Column 1 represents real events > 0 (Positive real event).
    tp=len(pandas_top[(pandas_top[0]==True) & (pandas_top[1]==True)])  # True Positives: Predicted > 0 AND Real > 0
    tn=len(pandas_top[(pandas_top[0]==False) & (pandas_top[1]==False)])  # True Negatives: Predicted == 0 AND Real == 0
    fp=len(pandas_top[(pandas_top[0]==True) & (pandas_top[1]==False)])  # False Positives: Predicted > 0 AND Real == 0
    fn=len(pandas_top[(pandas_top[0]==False) & (pandas_top[1]==True)])  # False Negatives: Predicted == 0 AND Real > 0

    # Create the confusion matrix NumPy array.
    confusion_matrix_m=np.array([[tp, fp], [fn, tn]])
    # confusion_matrix_m # Implicit output in original code (kept as comment)

    # Calculate Precision and Accuracy, handling potential division by zero.
    try:
        # Precision: TP / (TP + FP) - proportion of positive predictions that were correct.
        precision=confusion_matrix_m[0][0]/(confusion_matrix_m[0][0]+confusion_matrix_m[0][1])
        # Accuracy: (TP + TN) / Total - proportion of correct predictions overall.
        accuracy=(confusion_matrix_m[0][0]+confusion_matrix_m[1][1])/(confusion_matrix_m[0][0]+confusion_matrix_m[0][1]+confusion_matrix_m[1][0]+confusion_matrix_m[1][1])
    except:
        # If division by zero occurs (e.g., no positive predictions for precision), set metrics to None.
        precision=None
        accuracy=None

    # Return the confusion matrix, precision, and accuracy.
    return confusion_matrix_m, precision, accuracy
    # Commented out original return values for f_benefit_k_most:
    # return abs(pd.DataFrame(M).T.sort_values(0,ascending=False).reset_index(drop=True).head(k).sum(axis=1)).mean()
    # return abs(pd.DataFrame(M).T.sort_values(1,ascending=True).reset_index(drop=True).head(k).sum(axis=1)/pd.DataFrame(M).T.sort_values(1,ascending=True).reset_index(drop=True).head(k)[1]).mean()


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

#AD Absolute difference
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

# Fairness functions lower is better  (more protected variables)

## square of difference between banefict function of two groups
def square_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        return (f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))**2
    except Exception as e:
        return e

## abs of difference between banefict function of two groups
def abs_diff_benefit(mask1,mask2,f_benefit,real_events,intensity_matrix,ben_kwgs={}):

    try:
        return abs(f_benefit(mask1,real_events,intensity_matrix,**ben_kwgs)-f_benefit(mask2,real_events,intensity_matrix,**ben_kwgs))
    except Exception as e:
        return e

## Mean of difference between banefict function of all pair of groups
def variance(dict_masks,f_benefit,real_events,intensity_matrix):
    try:
        return np.array([square_diff_benefit(dict_masks[i],dict_masks[j],f_benefit,real_events,intensity_matrix) for i in dict_masks.keys() for j in dict_masks.keys()]).mean()
    except Exception as e:
        return e

## Max difference between benefict function of all pair of groups
# This is a fairness measure that calculates the difference between the maximum and minimum
# benefit values observed across all defined groups, specifically using f_benefit_k_most.
# Lower is better. Can also calculate the Ratio (higher is better).
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use (not directly used here, f_benefit_k_most is used).
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   k_p_up (dict or similar): A dictionary or structure providing the 'k' value for each group mask.
#   measure (str): "Diff" for Max-Min difference (default), "Ratio" for Min/Max ratio.
# Returns:
#   float: The calculated Max-Min difference or Min/Max ratio using f_benefit_k_most.
# Raises:
#   Exception: If an error occurs during calculation.
def MM(dict_masks,f_benefit,real_events,intensity_matrix,k_p_up,measure="Diff"):
    try:
        # Calculate the benefit for each mask using f_benefit_k_most with the specific k value for that group.
        # NOTE: f_benefit_k_most now returns a tuple (confusion matrix, precision, accuracy).
        # This list comprehension will collect tuples. The MM calculation below will need to handle this.
        R=[f_benefit_k_most(dict_masks[i],real_events,intensity_matrix,k_p_up[i]) for i in dict_masks]
        # Assuming MM should be calculated based on one of the returned values, e.g., precision (index 1).
        # Extract precision values from the list of tuples.
        precision_values = [result[1] for result in R if result[1] is not None] # Handle None precision

        if not precision_values: # Handle case where all precisions are None
             return None

        if measure == "Diff":
            # Return the difference between the maximum and minimum precision values.
            return max(precision_values)-min(precision_values)
        if measure == "Ratio":
            # Return the ratio of the minimum to the maximum precision values (higher is better).
            # Handle potential division by zero if max(precision_values) is 0.
            max_precision = max(precision_values)
            if max_precision == 0:
                return None # Or some other indicator for undefined ratio
            return min(precision_values)/max_precision

    except Exception as e:
        return e # Return the exception if calculation fails

## Calculate the Gini coefficient.
# This is a measure of statistical dispersion, here applied to the distribution of benefit
# values across groups, specifically using f_benefit_k_most. Lower is better.
# Args:
#   dict_masks (dict): A dictionary where keys are group names and values are their boolean masks.
#   f_benefit (function): The benefit function to use (not directly used here, f_benefit_k_most is used).
#   real_events (np.ndarray): Real event grid.
#   intensity_matrix (np.ndarray): Predicted intensity grid.
#   k_p_up (dict or similar): A dictionary or structure providing the 'k' value for each group mask.
# Returns:
#   float: The calculated Gini coefficient using values from f_benefit_k_most (likely precision or accuracy).
# Raises:
#   Exception: If an error occurs during calculation.
def gini_coeff(dict_masks,f_benefit,real_events,intensity_matrix,k_p_up):
    try:
        # Calculate the benefit for each mask using f_benefit_k_most with the specific k value for that group.
        # NOTE: f_benefit_k_most now returns a tuple (confusion matrix, precision, accuracy).
        # This list comprehension will collect tuples. The Gini calculation below will need to handle this.
        R_tuples=[f_benefit_k_most(dict_masks[i],real_events,intensity_matrix,k_p_up[i]) for i in dict_masks]

        # Assuming Gini should be calculated based on one of the returned values, e.g., precision (index 1).
        # Extract precision values from the list of tuples, filtering out None values.
        R = np.array([result[1] for result in R_tuples if result[1] is not None])

        if R.size == 0 or R.sum() == 0: # Handle cases with no valid precision values or sum is zero
             return None # Or some other indicator for undefined Gini

        # R=1-R/R.sum() # Commented out line

        # Calculate the denominator for the Gini coefficient formula.
        den=2*len(R)*sum(R)
        # Calculate the Gini coefficient using the formula.
        G=abs(R.reshape(len(R),1)-R.reshape(1,len(R))).sum()/den
        return G # Return the calculated Gini coefficient
    except Exception as e:
        return e # Return the exception if calculation fails


## Calculate the Jain's fairness index (higher is better).
# This index measures the "fairness" of a set of values, here applied to the distribution
# of benefit values across groups. A value of 1 indicates perfect fairness.
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
        # Calculate the benefit for each mask using the standard f_benefit function and convert to a NumPy array.
        R=np.array([f_benefit(dict_masks[i],real_events,intensity_matrix) for i in dict_masks])
        # R=1-R/R.sum() # Commented out line

        # Calculate the denominator for Jain's index formula.
        den=len(R)*sum(R**2)
        # Calculate Jain's index using the formula.
        # Handle potential division by zero if den is zero.
        if den == 0:
             return None # Or some other indicator for undefined Jain index
        J=(R.sum())**2/den # Corrected formula from original code (was **1)
        return J # Return the calculated Jain's index
    except Exception as e:
        return e # Return the exception if calculation fails
