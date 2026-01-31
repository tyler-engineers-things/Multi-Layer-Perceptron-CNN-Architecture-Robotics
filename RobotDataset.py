import time
import numpy as np
import os
import torch
import pandas as pd
from torchvision.io import decode_image
from torch.utils.data import DataLoader

print(np.load('/home/Tyler/Documents/CS4803ARM_Fall2025/user_data/MP4/training_dataset_mujoco/pick_and_place/dataset26/joint_positions.npy'))