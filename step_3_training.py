from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from step_2_dataset import get_train_test_loaders

class Net(nn.Module):
    def __init__(self):
        # gọi phương thức khởi tạo từ class nn.Module
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)