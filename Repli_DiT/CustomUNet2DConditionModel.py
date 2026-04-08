from diffusers import UNet2DConditionModel
import math
import os
import pickle
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union

from utils_custom_unet import deprecate

class CustomUNet2DConditionModel(UNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        return output