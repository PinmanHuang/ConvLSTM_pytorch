from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np
import torch

class MyDataset(Dataset):
    def __init__(self, dir, history_window=12, future_window=12, mode='train', split_ratio=0.9, transform=None, target_transform=None):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.dir = dir
        self.history_window = history_window
        self.future_window = future_window
        self.mode = mode
        self.split_ratio = split_ratio
        self.transform = transform
        self.target_transform = target_transform
        # Load all dataset
        self.all_tensor_data = None
        for file in Path(self.dir).iterdir():
            try:
                # Load np data
                np_matrix = np.load(file)
                # Append the np_matrix_transform to 3D Tensor
                tensor_data = torch.tensor(np_matrix, dtype=torch.float)
                tensor_data = tensor_data.view(1, 1, np_matrix.shape[0], np_matrix.shape[1])
                if self.all_tensor_data is None:
                    self.all_tensor_data = tensor_data
                else:
                    self.all_tensor_data = torch.cat([self.all_tensor_data, tensor_data])
            except Exception:
                continue
        self.imgs = self.split_data(self.split_ratio, self.mode)
        
            
    def split_data(self, ratio, mode):
        train_len = int(self.all_tensor_data.shape[0]*ratio)
        test_len = self.all_tensor_data.shape[0]-train_len
        train_dataset, test_dataset = torch.split(self.all_tensor_data, [train_len, test_len])
        if mode == 'train':
            return train_dataset
        else:
            return test_dataset

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. 
        # --------------------------------------------
        x = self.imgs[index:index+self.history_window]
        y = self.imgs[index+self.history_window:index+self.history_window+self.future_window]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs) - self.history_window - self.future_window