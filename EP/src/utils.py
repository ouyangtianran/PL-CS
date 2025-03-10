from torch.utils.data.dataloader import default_collate
import numpy as np

def get_collate(name):
    if name == "identity":
        return lambda x: x
    else:
        return default_collate  
    
def shrink_label(list, label):
    for i, l in enumerate(label):
        ind = np.where(list == l)
        list[ind] = i
    return list
