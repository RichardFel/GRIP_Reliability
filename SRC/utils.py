import numpy as np
import os


def check_or_create_folder(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If it does not exist, create the folder
        os.makedirs(folder_path)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def probability(vector):
    p_vector = vector / sum(vector)
    return p_vector


def entropy(p_vector):
    p_0 = p_vector[0]*np.log(p_vector[0])
    for p in p_vector[1:]:
        p_0 = p_0 + p*np.log(p)
    return -p_0
