import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
import numpy as np
import random
import os
import pandas as pd
import torch.nn as nn
from collections import defaultdict

from scipy.integrate import simps
import pickle
import time

import sys
sys.path.append("..")

import tools.utils as utils
from tools.small_model import FC_MD
from RicciCurvature.OllivierRicci import OllivierRicci
from tools.FC_linear import FC_Linear
# from RicciCurvature.q_exponential import q_exponential

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
