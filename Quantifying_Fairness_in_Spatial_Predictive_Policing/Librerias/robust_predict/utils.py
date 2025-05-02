# spatial_simulation_utils.py
#
# This file contains utility functions for spatial operations, data transformation,
# simulation, and optimization related to crime prediction and fairness analysis.

import numpy as np # For numerical operations
import matplotlib.pyplot as plt # For plotting
import sparse as sp # For working with sparse arrays (used for data representation)

import multiprocessing # For parallel processing (imported, but commented out usage)
from multiprocessing import Pool # For managing a pool of worker processes (imported, but commented out usage)
coreCount = multiprocessing.cpu_count() # Get the number of CPU cores (imported, but commented out usage)
#pool = max(Pool(coreCount - 4) ,1) # Commented out: Initialize a process pool


##### np.array (Comment indicating a section related to NumPy arrays)

# Define a function to plot a 2D intensity matrix.
# Args:
#   intensity (np.ndarray): A 2D NumPy array representing intensity values on a grid.
#   vmax (float, optional): The maximum value for the colormap scaling. Defaults to None.
def plot_intensity(intensity,vmax=None):
    # Create a figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(11,5))
    # Display the intensity matrix as an image using the 'jet' colormap.
    # interpolation='none' prevents interpolation between pixels. alpha sets transparency. vmin=0 sets minimum color value.
    # vmax sets the maximum color value for scaling.
    im=ax.imshow(intensity, 'jet', interpolation='none', alpha=0.7,vmin=0,vmax=vmax)
    # Invert the y-axis to match typical image/grid coordinates (origin top-left).
    ax.invert_yaxis()
    # Add a color bar to the plot to show the mapping of values to colors.
    plt.colorbar(im)
    # Display the plot.
    plt.show(ax)

### coord to index pos array (Comment indicating coordinate to index conversion)

# Define a function to convert spatial coordinates to grid cell indices.
# Args:
#   coords (np.ndarray): A 2D NumPy array where the first row is x-coordinates and the second row is y-coordinates.
#   size_m (tuple): A tuple (rows, columns) representing the dimensions of the grid.
# Returns:
#   np.ndarray: A 2D NumPy array where the first row is row indices and the second row is column indices in the grid.
def get_position(coords,size_m):
    # Transpose coordinates, flip order (y, x), multiply by grid size, and floor to get integer indices.
    # NOTE: This assumes coordinates are normalized between 0 and 1 and size_m is (rows, columns).
    pos=np.floor(coords[::-1].T*np.array(size_m)).astype(int)
    # Handle edge cases where a coordinate falls exactly on the upper boundary,
    # assigning it to the last row/column instead of out of bounds.
    pos[:,0][pos[:,0]==size_m[0]]=size_m[0]-1 # Cap row index at max row index
    pos[:,1][pos[:,1]==size_m[1]]=size_m[1]-1 # Cap column index at max column index
    return pos.T # Return transposed indices (row, column)


### define neigborhood cells around (Comment indicating neighborhood definition)

# Define a function to get the indices of neighboring cells around a given position within a mask.
# Considers 8 adjacent cells (including diagonals) and the cell itself.
# Args:
#   posicion (tuple): A tuple (row, column) representing the central cell's indices.
#   mask (np.ndarray): A boolean mask representing the valid grid cells.
# Returns:
#   list: A list of tuples, where each tuple is the (row, column) index of a valid neighboring cell.
def neighborhood_get(posicion,mask):
    ## posicion (x,y) - NOTE: Comment says (x,y), but code uses (row, column) indices.
    filas, columnas = mask.shape # Get mask dimensions
    fila, columna = posicion # Unpack central cell position

    celdas_adyacentes = [] # Initialize list for neighboring cells

    # Iterate through a 3x3 window centered at the given position.
    for i in range(fila-1, fila+2):
        for j in range(columna-1, columna+2):
            # Check if the current cell (i, j) is within the mask boundaries
            # AND if the cell is True in the provided mask.
            if 0 <= i < filas and 0 <= j < columnas and mask[i, j]:
                celdas_adyacentes.append((i, j)) # Add the valid neighboring cell index

    return celdas_adyacentes # Return the list of valid neighboring cell indices


### pos matrix to flatten (Comment indicating flattening position matrix)

# Define a function to flatten a matrix of 2D grid positions into 1D indices.
# Args:
#   pos_matrix (np.ndarray): A 2D NumPy array where the first row is row indices and the second row is column indices.
#   size_m (tuple, optional): A tuple (rows, columns) representing the dimensions of the grid. Defaults to (5, 5).
# Returns:
#   np.ndarray: A 1D NumPy array of flattened indices.
def get_flatten_pos(pos_matrix,size_m=(5,5)):
    # np.ravel_multi_index converts a tuple of coordinate arrays to a flat index array.
    # Here, pos_matrix[0] are row indices and pos_matrix[1] are column indices.
    return np.ravel_multi_index(pos_matrix, size_m)


#### create W as identifier cell (Comment indicating creation of W matrix)

# Define a function to create a sparse matrix W, likely representing covariates for each data point.
# W seems to encode the grid cell position for each event.
# Args:
#   filas (int): Number of rows in the spatial grid.
#   columnas (int): Number of columns in the spatial grid.
#   data_shape (tuple): The shape of the input data (e.g., (time_steps, rows, columns)).
#   data_size (int): The total number of non-zero data points (events).
# Returns:
#   sp.COO: A sparse COO matrix encoding grid cell identifiers for each event.
def create_W(filas,columnas,data_shape,data_size):
    ### W must have the size like a vector for each point in data (Comment explaining W's intended structure)

    m=filas*columnas ## number of covariates (Comment defining m as total grid cells)

    #### covariates // for identify cell position (Comment explaining the purpose of covariates)
    # Create coordinate arrays for a 3D grid (e.g., time, row, column).
    new_coors=np.indices(data_shape).reshape((3,data_size))

    # Define a lambda function to get the flattened 1D index from 2D (row, column) indices.
    aux_func = lambda x : get_flatten_pos(x,size_m=(filas,columnas))

    # Apply the lambda function to the transposed row/column coordinates to get flattened indices.
    newrow = np.array(list(map(aux_func,new_coors[1:,:].T)))
    # Stack the original 3D coordinates with the new flattened 1D indices.
    new_coors = np.vstack([new_coors, newrow])

    # Create a sparse COO matrix.
    # The coordinates are new_coors (time, row, column, flattened_index).
    # The data values are all 1s (indicating the presence of an event at that coordinate).
    # The shape is the original data shape plus the number of covariates (flattened grid size).
    return sp.COO(new_coors,np.ones(new_coors.shape[1]),data_shape+(m,))


# Define a function to add an independent covariate (column of ones) to the sparse matrix W.
# Args:
#   W (sp.COO): The input sparse COO matrix W.
# Returns:
#   sp.COO: A new sparse COO matrix with an additional column of ones.
def add_independient_to_w(W):
    # Convert the sparse matrix to a dense NumPy array.
    W_=W.todense()
    # Append a column of ones along the last axis (axis=3), reshaping it to match the shape of W_.
    W_=np.append(W_, np.ones(W.shape[:-1]).reshape(W.shape[:-1]+(1,)), axis=3)
    # Convert the resulting dense array back to a sparse COO matrix.
    return sp.COO(W_)


# Define a function to create a sparse matrix W with continuous covariates, likely for regression.
# This version seems to encode the normalized position within the grid.
# Args:
#   filas (int): Number of rows in the spatial grid.
#   columnas (int): Number of columns in the spatial grid.
#   data_shape (tuple): The shape of the input data (e.g., (time_steps, rows, columns)).
#   data_size (int): The total number of non-zero data points (events).
# Returns:
#   sp.COO: A sparse COO matrix encoding normalized continuous spatial covariates.
def create_W_continuo(filas,columnas,data_shape,data_size):
    ### W must have the size like a vector for each point in data (Comment explaining W's intended structure)

    m=filas*columnas ## number of covariates (Comment defining m as total grid cells)
    # Create coordinate arrays for a 3D grid (time, row, column).
    new_coors=np.indices(data_shape,).reshape((3,data_size))
    # Define a lambda function to get the flattened 1D index from 2D (row, column) indices.
    aux_func = lambda x : get_flatten_pos(x,size_m=(filas,columnas))
    # Apply the lambda function to get flattened indices and normalize them by the total number of cells.
    positions = np.array(list(map(aux_func,new_coors[1:,:].T))) / m
    # Create a new coordinate array structure for the sparse matrix.
    # This structure seems designed to pair original coordinates with a continuous value and an indicator.
    new_coors2=np.zeros((4,new_coors.shape[1]*2))
    new_coors2[:3,::2]=new_coors # Copy original 3D coordinates to even columns
    new_coors2[:3,1::2]=new_coors # Copy original 3D coordinates to odd columns
    new_coors2[-1,1::2]=1 # Set the last row to 1 for odd columns (indicator)

    # Create the data values array for the sparse matrix.
    newrow = np.ones(2*len(positions)) # Initialize with ones
    newrow[::2]=positions # Assign normalized positions to even columns

    # Create a sparse COO matrix.
    # The coordinates are new_coors2.
    # The data values are newrow.
    # The shape is the original data shape plus 2 (for the continuous value and indicator).
    return sp.COO(new_coors2.astype(int),newrow,shape=data_shape+(2,))



###################################
## heuristic function (Comment indicating a section for heuristic functions)

# Define a heuristic rule function, likely for calculating a score or value based on covariates and parameters.
# Args:
#   data: Input data (not used in the function body).
#   thetha (np.ndarray): A NumPy array of parameters (weights).
#   W (sp.COO or np.ndarray): A matrix of covariates.
# Returns:
#   np.ndarray: The result of matrix multiplication between W and thetha.
def heuristic_rule(data,thetha,W):
    # Perform matrix multiplication (W @ thetha).
    # This assumes W and thetha are compatible for multiplication.
    return (W @ thetha)


##################################


# Define a function to calculate the likelihood for a specific cell at a specific time.
# Args:
#   time (int): The time index.
#   pos (tuple): A tuple (row, column) representing the cell's position.
#   thetha (np.ndarray): Parameters (weights).
#   data (sp.COO or np.ndarray): The data matrix.
#   W (sp.COO or np.ndarray): The covariate matrix.
#   likelihood_func (function): The likelihood function to apply.
# Returns:
#   float: The calculated likelihood value.
def likelihood_cell_time(time,pos,thetha,data,W,likelihood_func):
    x,y=pos # Unpack cell position

    # Calculate the likelihood using the provided likelihood function.
    # Passes the data value at the specific time and position (data[time,x,y]+1),
    # the parameters (thetha), and the covariate vector for that time and position (W[time,x,y]).
    return likelihood_func(data[time,x,y]+1,thetha,W[time,x,y])


# Define a function to get the index of the best neighboring cell based on likelihood.
# Designed to be used in parallel processing.
# Args:
#   inputs (tuple): A tuple containing:
#                   - time (int): The time index.
#                   - pos (tuple): The original cell's position (row, column).
#                   - thetha (np.ndarray): Parameters.
#                   - data (sp.COO or np.ndarray): The data matrix.
#                   - neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
#                   - W (sp.COO or np.ndarray): The covariate matrix.
#                   - likelihood_func (function): The likelihood function.
# Returns:
#   int: The index of the best neighboring cell in the neighborhood list (index corresponding to the minimum likelihood).
def get_best_change_cell_time(inputs):

    # Unpack the input tuple.
    time,pos,thetha,data,neighborhood,W,likelihood_func = inputs

    eval_like=[] # Initialize list to store likelihood values for neighbors
    # Iterate through each neighboring position for the given cell.
    for n_pos in neighborhood[pos]: # NOTE: Variable name changed from 'pos' to 'n_pos' for clarity
        # Calculate the likelihood for the current neighboring cell at the given time.
        eval_like.append(likelihood_cell_time(time,n_pos,thetha,data,W,likelihood_func)) # Use n_pos

    # Return the index of the minimum likelihood value in the list.
    # This index corresponds to the position of the best neighboring cell in the neighborhood list.
    return eval_like.index(min(eval_like))


# Define a function to get the best shift (movement) for each event based on likelihood.
# Iterates through cells with events and finds the best neighbor for each event's time and position.
# Args:
#   thetha (np.ndarray): Parameters (weights).
#   data (sp.COO or np.ndarray): The data matrix (time, row, column).
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
#   W (sp.COO or np.ndarray): The covariate matrix.
#   likelihood_func (function): The likelihood function.
# Returns:
#   tuple: A tuple of integers, where each integer is the index of the best neighboring cell
#          in the neighborhood list for a corresponding event.
def get_best_shift(thetha,data,neighborhood,W,likelihood_func):

    coding=[] # Initialize list to store the best shift index for each event
    # Iterate through each cell position that is part of the neighborhood dictionary keys.
    for pos in list(neighborhood.keys()):

        try:
            # Attempt to get time indices of non-zero values (events) in the current cell using sparse coords.
            index_data=data[:,pos[0],pos[1]].coords.flatten()
        except:
            # If sparse coords not available, use np.where to find non-zero indices.
            index_data=np.where(data[:,pos[0],pos[1]]>0)[0]

        # Iterate through each time index where an event occurred in the current cell.
        for time in index_data:
            # Get the index of the best neighboring cell for this event's time and position.
            best_neighbor_index = get_best_change_cell_time([time,pos,thetha,data,neighborhood,W,likelihood_func])
            # Append the best neighbor index to the coding list for each event that occurred at this time and position.
            code = [best_neighbor_index] * int(data[:,pos[0],pos[1]][time])
            coding+=code

    return tuple(coding) # Return the list of best neighbor indices as a tuple

# Commented out original parallel processing version of get_best_shift:
# def get_best_shift(thetha,data,neighborhood,W,likelihood_func):
#     coding=[]
#     for pos in list(neighborhood.keys()):
#         try:
#             index_data=data[:,pos[0],pos[1]].coords.flatten()
#         except:
#             index_data=np.where(data[:,pos[0],pos[1]]>0)[0]
#         inputs_=[[x,pos,thetha,
#                 data,neighborhood,
#                 W,likelihood_func] for x in index_data]
#         results = pool.map(get_best_change_cell_time, inputs_)
#         for idt,time in enumerate(index_data):
#             code= [results[idt]]*int(data[:,pos[0],pos[1]][time])
#             coding+=code
#     return tuple(coding)

# Define a function to get the "no shift" coding for events.
# This assumes each event stays in its original cell.
# Args:
#   data (sp.COO or np.ndarray): The data matrix (time, row, column).
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
# Returns:
#   tuple: A tuple of integers, where each integer is the index of the original cell
#          within its own neighborhood list for a corresponding event.
def get_no_shift(data,neighborhood):
    coding=[] # Initialize list to store the "no shift" index for each event
    # Iterate through each cell position that is part of the neighborhood dictionary keys.
    for pos in list(neighborhood.keys()):
        try:
            # Attempt to get time indices of non-zero values (events) in the current cell using sparse coords.
            index_data=data[:,pos[0],pos[1]].coords.flatten()
        except:
            # If sparse coords not available, use np.where to find non-zero indices.
            index_data=np.where(data[:,pos[0],pos[1]]>0)[0]
        # Iterate through each time index where an event occurred in the current cell.
        for time in index_data:
            # Find the index of the original cell 'pos' within its own neighborhood list.
            original_cell_index_in_neighborhood = (neighborhood[pos]).index(pos)
            # Append this index to the coding list for each event that occurred at this time and position.
            code = [original_cell_cell_index_in_neighborhood] * int(data[:,pos[0],pos[1]][time])
            coding+=code
    return tuple(coding) # Return the list of indices as a tuple


# Define a function to perform the spatial shift on the data based on a given shift coding.
# Creates a new data matrix where events are moved according to the shift.
# Args:
#   shift (tuple): A tuple of integers representing the destination neighbor index for each event.
#   data (sp.COO or np.ndarray): The original data matrix (time, row, column).
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
# Returns:
#   sp.COO or np.ndarray: A new data matrix with events shifted according to the 'shift' coding.
def do_shift(shift,data,neighborhood):

    try :
        # Attempt to initialize a new sparse DOK matrix with the same shape as the input data.
        new_data=sp.DOK(data.shape)
    except:
        # If sparse DOK initialization fails, initialize a dense NumPy array of zeros.
        new_data=np.zeros_like(data)

    total_events=0 # Initialize a counter for the total number of events processed so far

    #####
    # Iterate through each cell position that is part of the neighborhood dictionary keys.
    for pos in list(neighborhood.keys()):
        # Get the total number of events in the current cell across all time steps.
        events_in_pos= int(data[:,pos[0],pos[1]].sum())

        # Extract the shift coding for the events in the current cell from the overall 'shift' tuple.
        new_pos_indices=shift[total_events:total_events+events_in_pos]

        # Update the total events counter.
        total_events+=events_in_pos

        # Get time indices where events occurred in the current cell.
        index_data=np.where(data[:,pos[0],pos[1]]>0)[0]

        evento=0 # Counter for events within the current cell
        # Iterate through each time index where an event occurred.
        for time in index_data:
            # Iterate for the number of events that occurred at this specific time and position.
            for _ in range(int(data[:,pos[0],pos[1]][time])):
                # Get the new spatial position (row, column) based on the shift index and the neighborhood list.
                new_x,new_y=neighborhood[pos][new_pos_indices[evento]]
                # Increment the count in the new_data matrix at the shifted position and original time.
                new_data[time,new_x,new_y]+=1
                evento+=1 # Increment event counter within the cell

    return new_data # Return the data matrix with shifted events


############# (Comment indicating a new section)

# Define a function for Backtracking Line Search, an optimization algorithm step.
# Finds a suitable step size (alpha) that satisfies the Armijo condition.
# Args:
#   alpha (float): Initial step size.
#   xk (np.ndarray): Current parameters.
#   pk (np.ndarray): Search direction (e.g., negative gradient).
#   f (function): The objective function to minimize.
#   gradf (function): The gradient of the objective function.
#   rho (float, optional): Reduction factor for alpha (0 < rho < 1). Defaults to 0.8.
#   c (float, optional): Parameter for the Armijo condition (0 < c < 1). Defaults to 0.3.
# Returns:
#   float: The calculated step size (alpha).
def BacktrackingLineSearch(alpha,xk,pk,f,gradf,rho=0.8,c=0.3):
  # while f(xk + alpha * pk) >= f(xk) + c * alpha * (gradient of f at xk * pk):
  while f(xk+alpha*pk)>=f(xk)+c*alpha*(gradf(xk)@pk):
    alpha*=rho # Reduce alpha
    # print(f'paso={alpha},iteración={count}' # Commented out: Debug print
  return alpha # Return the step size


###### Prepare data (Comment indicating a section for data preparation)

# Define a function to transform and scale a vector to the range [0, 1].
# Args:
#   vect (np.ndarray): The input NumPy array (vector).
# Returns:
#   np.ndarray: The scaled vector.
def transform_scale(vect):
    # Define the original range [min, max] and the target range [0, 1].
    x=[vect.min(),vect.max()]
    y=[0,1]
    # Fit a polynomial of degree 1 (linear) to map the original range to the target range.
    coefficients = np.polyfit(x, y, 1)
    # Create a polynomial object from the coefficients.
    polynomial = np.poly1d(coefficients)
    # Apply the polynomial to the input vector to perform the scaling.
    return polynomial(vect)


# Define a function to convert open_cp.TimedPoints data to a sparse 3D data matrix.
# Args:
#   time_points (open_cp.TimedPoints): The input spatio-temporal point data.
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
#   filas (int): Number of rows in the spatial grid.
#   columnas (int): Number of columns in the spatial grid.
# Returns:
#   sp.COO: A sparse COO matrix representing event counts in a 3D (time, row, column) grid.
def time_points_to_data_sparse(time_points,neighborhood,filas,columnas):
    # Transform and scale the x and y coordinates of the time points to the range [0, 1].
    # NOTE: This modifies the original time_points object's coords in-place.
    time_points.coords[0] = transform_scale(time_points.coords[0])
    time_points.coords[1] = transform_scale(time_points.coords[1])

    # Convert the scaled coordinates to grid cell indices.
    # Assumes scaled coordinates are between 0 and 1 and (filas, columnas) is the grid size.
    data_posiciones=get_position(time_points.coords,(filas,columnas))
    # Get the minimum year from the timestamps.
    min_year=time_points.times_datetime().min().year
    # Calculate the number of days since the beginning of the minimum year for each timestamp.
    days_step=[(i.year-min_year)*365+i.timetuple().tm_yday for i in time_points.times_datetime()]

    # Create a dictionary to map unique day steps to sequential integer indices.
    dict_change={}
    for idx,i in enumerate(sorted(set(days_step))):
        dict_change[i]=idx

    # Calculate the total number of time steps (unique days).
    time_steps=max(dict_change.values())

    # Convert the day steps to the sequential integer indices.
    days_step=[dict_change[i] for i in days_step]

    # Combine the sequential time steps with the grid position indices.
    # data_procc will have shape (3, num_events) where rows are [time_index, row_index, col_index].
    data_procc=np.concatenate((np.array(days_step).reshape(1,data_posiciones.shape[1]),data_posiciones))

    #############################################
    # Create a dense NumPy array to count events per cell per time step.
    # Shape is (time_steps + 1, rows, columns).
    data_procc2=np.zeros((time_steps+1,filas,columnas))

    # Iterate through each time step.
    for t in range(time_steps+1):
        # Filter the processed data points to get only those belonging to the current time step.
        masked_time=data_procc.T[data_procc[0]==t].T

        # Create a temporary array of zeros with the same spatial dimensions as the grid.
        temp_count_ev=np.zeros_like(data_procc2[t])

        # Iterate through each cell position that is part of the neighborhood dictionary keys.
        for pos in neighborhood.keys():

            # Create a boolean mask to identify points in masked_time that match the current cell position.
            # Summing the equality check across axis=1 == 2 means both row and column indices match.
            mask_pos=(masked_time[1:,:].T == np.array(pos)).sum(axis=1) == 2

            # Count the number of points that match the current cell position and store in temp_count_ev.
            temp_count_ev[pos]=len(masked_time.T[mask_pos])

        # Assign the event counts for the current time step to the corresponding slice in data_procc2.
        data_procc2[t]=temp_count_ev
    #############################################
    # Convert the dense event count array to a sparse COO matrix.
    data_sparse = sp.COO(data_procc2)

    return data_sparse # Return the sparse 3D data matrix


# Define a function to generate new data by probabilistically moving events based on a 'psi' value.
# Events in cells with data_sparse.data > 0 are moved to a random neighbor with probability 'psi'.
# Args:
#   data_sparse (sp.COO): The input sparse 3D data matrix.
#   psi (float): The probability of an event moving to a neighbor.
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
# Returns:
#   sp.COO: A new sparse 3D data matrix with some events potentially moved.
def gen_data_from_sparse(data_sparse,psi,neighborhood):

    # For each non-zero entry (event count) in data_sparse, determine how many events to move
    # using a binomial distribution with n=event_count and p=psi.
    to_move=np.random.binomial(data_sparse.data.astype(int),psi)
    # Create a sparse COO matrix representing the number of events to move from each original location.
    to_move=sp.COO(data_sparse.coords,to_move,data_sparse.shape)

    # Initialize a dense NumPy array of zeros to store the moved events.
    # NOTE: Using a dense array here might be memory-intensive for large grids/times.
    moved=np.zeros(data_sparse.shape)
    # Iterate through the non-zero entries (events to move) in the 'to_move' sparse matrix.
    for events_to_move,original_coords in zip(to_move.data,to_move.coords.T): # Iterate through data and corresponding coordinates
        o_pos=tuple(original_coords[1:]) # Get the original spatial position (row, column)
        # For each event to move from this original position:
        for event in range(events_to_move):
            # Randomly select a new neighboring position from the neighborhood list for the original position.
            new_pos=neighborhood[o_pos][np.random.choice(len(neighborhood[o_pos]))]
            # Increment the count in the 'moved' array at the original time step and the new spatial position.
            moved[original_coords[0],new_pos[0],new_pos[1]]+=1

    # Convert the dense 'moved' array to a sparse COO matrix.
    moved = sp.COO(moved)

    # Calculate the new data matrix: original data minus events moved out, plus events moved in.
    data_moved=data_sparse-to_move+moved

    return data_moved # Return the new data matrix


# Define a function to generate simulated data based on a hot spot pattern and movement probability.
# Simulates events over time, with a probability 'psi' of moving from the current hot spot to a random neighbor.
# The hot spot is determined by the cell with the maximum mean intensity over a lookback window T.
# Args:
#   psi (float): The probability of an event moving from the hot spot.
#   neighborhood (dict): A dictionary mapping cell positions to lists of neighboring cell positions.
#   T (int): The lookback window size (number of previous days to average for hot spot).
#   T0 (np.ndarray): The initial intensity pattern (e.g., for the first day).
#   total_days (int): The total number of days to simulate.
#   mean_daily (float): The expected average number of events per day.
# Returns:
#   sp.COO: A sparse COO matrix representing the simulated event counts over time and space.
def gen_data_from_T0(psi,neighborhood,T,T0,total_days,mean_daily):
    # Initialize a dense NumPy array of zeros to store the simulated data.
    # Shape is (total_days, rows, columns) based on T0's spatial shape.
    data=np.zeros((total_days,)+T0.shape)

    # Generate the number of events for each day using a Poisson distribution with the specified mean.
    events_by_time=np.random.poisson(mean_daily,total_days)

    # total_days (Comment indicating the loop is over total_days)
    # Iterate through each day of the simulation.
    for i in range(total_days):
        if i==0:
            # For the first day, use the initial intensity pattern T0 to determine the hot spot.
            compare=T0
        else:
            # For subsequent days, calculate the mean intensity over the lookback window T to determine the hot spot.
            compare=data[max(0,i-T):i].mean(axis=0)

        # Find the indices (row, column) of the cell with the maximum intensity (the hot spot).
        hot_spot=np.unravel_index(compare.argmax(), compare.shape)

        # Simulate events for the current day based on the generated number of events.
        for event in range(events_by_time[i]):

            rand_=np.random.rand() # Generate a random number between 0 and 1
            if rand_ < psi:
                # If the random number is less than psi, move the event to a random neighbor of the hot spot.
                new_pos=neighborhood[hot_spot][np.random.choice(len(neighborhood[hot_spot]))]
            else:
                # Otherwise, the event stays in the hot spot cell.
                new_pos=hot_spot

            # Increment the count in the 'data' array at the current day and the determined new spatial position.
            data[i,new_pos[0],new_pos[1]]+=1

    # Convert the dense simulated data array to a sparse COO matrix.
    return  sp.COO(data) # Return the sparse simulated data matrix
