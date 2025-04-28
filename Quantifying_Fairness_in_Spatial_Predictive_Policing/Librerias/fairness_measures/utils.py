import numpy as np
import pandas as pd

#### Protected variables

#### binary

### random with probability individual for cell
def get_random_matrix_bool(rows,columns,p=0.5):
    return np.random.choice(a=[False, True], size=(rows, columns), p=[p, 1-p])

### dict of patterns of mask
def patterns_masks(rows,columns):
    masks={}
    ones=np.ones((rows,columns)).astype(bool)

    masks["0"]=ones.copy()
    masks["0"][:,:int(columns/2)]=False

    masks["1"]=~masks["0"]

    masks["2"]=ones.copy()
    masks["2"][:int(rows/2),:]=False

    masks["3"]=~masks["2"]

    masks["4"]=ones.copy()
    masks["4"][:int(rows/2),:int(columns/2)]=False
    masks["4"][int(rows/2):,int(columns/2):]=False

    masks["5"]=~masks["4"]

    masks["6"]=np.tril(ones)
    masks["7"]=~masks["6"]

    masks["8"]=masks["6"][:,::-1]
    masks["9"]=~masks["8"]

    return masks
    
def compare_masks(mask1,mask2):
    return (mask1 == mask2).all()


def generate_total_masks(dict_mask):
    total_mask={}
    for key, mask in dict_mask.items():

        rep_= True in [compare_masks(mask,a_mask) for a_mask in total_mask.values()]

        if not rep_:
            total_mask[str(key)]=mask

        rep_= True in [compare_masks(~mask,a_mask) for a_mask in total_mask.values()]

        if not rep_:
            total_mask[str(key)+"_r"]=~mask

    return total_mask