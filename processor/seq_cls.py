import os
import copy
import json
import logging

import torch
from torch.utils.data import TensorDataset

seq_cls_tasks_num_labels = {
    "nsmc": 2
}

seq_cls_output_modes = {
    "nsmc": "classification"
}