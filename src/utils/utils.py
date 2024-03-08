import numpy as np
import torch
import torch.nn as nn

# ----- Random Seed Control -----

def fix_random_seed(seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
