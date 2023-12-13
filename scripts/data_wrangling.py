import seaborn as sns
import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import os
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, SubsetRandomSampler, random_split
import albumentations as album
import re
from torchvision.transforms import v2
from torchsummary import summary


# Helper functions
def resize(image):
    comp = album.Compose([
        album.Resize(500, 1000, p=1, always_apply=True)
    ])
    return comp(image=image)['image']
def identity(x):
    return x
def one_hot_transform(labels): 
    one_hot_encoded = torch.nn.functional.one_hot(labels, 5)
    return one_hot_encoded.to(dtype=torch.float32)

def numpy_and_binarize(x):
    return np.where(x.detach().cpu().numpy() > 0.5, 1, 0)


