# mask_utils.py
#
# This file contains utility functions for generating, manipulating, and comparing
# boolean masks, primarily intended for defining spatial regions on a grid.

import numpy as np
import pandas as pd # pandas is imported but not used in this snippet.

#### Protected variables (Comment indicates a section related to protected variables, but none are defined here) ####

#### binary (Comment indicates a section related to binary masks) ####

### Generate a random boolean matrix with a specified probability of True for each element.
# Args:
#   rows (int): The number of rows in the matrix.
#   columns (int): The number of columns in the matrix.
#   p (float): The probability of a cell being False (default is 0.5).
# Returns:
#   np.ndarray: A 2D NumPy array of boolean values.
def get_random_matrix_bool(rows,columns,p=0.5):
    # np.random.choice selects elements from 'a' with probability 'p'.
    # size specifies the shape of the output array.
    # p=[p, 1-p] sets the probabilities for False and True respectively.
    return np.random.choice(a=[False, True], size=(rows, columns), p=[p, 1-p])

### Generate a dictionary of predefined boolean mask patterns.
# Args:
#   rows (int): The number of rows for the masks.
#   columns (int): The number of columns for the masks.
# Returns:
#   dict: A dictionary where keys are pattern names (strings) and values are boolean NumPy arrays.
def patterns_masks(rows,columns):
    masks={} # Initialize an empty dictionary to store masks
    ones=np.ones((rows,columns)).astype(bool) # Create a boolean array of all True

    # Define various mask patterns based on simple spatial divisions
    masks["0"]=ones.copy() # Start with all True
    masks["0"][:,:int(columns/2)]=False # Set the left half to False (vertical split)

    masks["1"]=~masks["0"] # Inverse of mask "0" (right half is False)

    masks["2"]=ones.copy() # Start with all True
    masks["2"][:int(rows/2),:]=False # Set the top half to False (horizontal split)

    masks["3"]=~masks["2"] # Inverse of mask "2" (bottom half is False)

    masks["4"]=ones.copy() # Start with all True
    masks["4"][:int(rows/2),:int(columns/2)]=False # Set the top-left quadrant to False
    masks["4"][int(rows/2):,int(columns/2):]=False # Set the bottom-right quadrant to False (diagonal quadrants are False)

    masks["5"]=~masks["4"] # Inverse of mask "4" (top-right and bottom-left quadrants are False)

    masks["6"]=np.tril(ones) # Create a lower triangular mask
    masks["7"]=~masks["6"] # Inverse of mask "6" (upper triangular mask)

    masks["8"]=masks["6"][:,::-1] # Flip mask "6" horizontally (lower triangular flipped)
    masks["9"]=~masks["8"] # Inverse of mask "8"

    return masks # Return the dictionary of generated mask patterns

### Compare two boolean masks for exact equality.
# Args:
#   mask1 (np.ndarray): The first boolean mask.
#   mask2 (np.ndarray): The second boolean mask.
# Returns:
#   bool: True if the masks are identical, False otherwise.
def compare_masks(mask1,mask2):
    # (mask1 == mask2) performs element-wise comparison, resulting in a boolean array.
    # .all() checks if all elements in the resulting boolean array are True.
    return (mask1 == mask2).all()

### Generate a dictionary of unique masks and their reverse (inverted) versions from an input dictionary of masks.
# Avoids adding duplicate masks or their direct inverses if they already exist.
# Args:
#   dict_mask (dict): A dictionary of boolean masks.
# Returns:
#   dict: A dictionary containing unique masks and their unique inverted counterparts.
def generate_total_masks(dict_mask):
    total_mask={} # Initialize an empty dictionary for unique masks

    # Iterate through each key-mask pair in the input dictionary
    for key, mask in dict_mask.items():

        # Check if the current mask is already present in total_mask
        rep_= True in [compare_masks(mask,a_mask) for a_mask in total_mask.values()]

        # If the mask is not a duplicate, add it to total_mask
        if not rep_:
            total_mask[str(key)]=mask

        # Check if the inverse of the current mask is already present in total_mask
        rep_= True in [compare_masks(~mask,a_mask) for a_mask in total_mask.values()]

        # If the inverse mask is not a duplicate, add it to total_mask with '_r' suffix
        if not rep_:
            total_mask[str(key)+"_r"]=~mask

    return total_mask # Return the dictionary of unique masks and their inverses
