import os

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def change_tuple_element(t, dim, val):
    l = list(t)
    l[dim] = val
    return tuple(l)