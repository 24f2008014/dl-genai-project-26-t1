import torch
from torch.utils.data import get_worker_info
import random
import numpy as np

def seed_worker(worker_id):
    """Seed the RNG for a Pytorch worker."""
    worker_info = get_worker_info()
    seed = (torch.initial_seed()) % (2**32-1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset = worker_info.dataset
    dataset.rng = random.Random(seed)