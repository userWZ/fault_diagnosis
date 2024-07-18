import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

def to_tensor(sample):
    features, label = sample
    return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class BaseDataset(Dataset):
    
    def __init__(self, data_dir: str|list[str], data_length: int = 1024, stride: int = 1024, num_samples: int = None, 
                 transform=to_tensor, scaler: bool = False) -> None:
        super().__init__()
        
        self.data_dir = data_dir
        self.data_length = data_length
        self.stride = stride
        self.num_samples = num_samples
        self.transform = transform
        
        if scaler:
            self.scaler = StandardScaler()
            self.X = scaler.fit_transform(self.X)
            
        self.data = []
        self.label = []
        if isinstance(data_dir, str):
            self.load_data(data_dir)
        else:
            for dir in data_dir:
                self.load_data(dir)
        
    def __getitem__(self, idx: int):    
        sample = self.data[idx], self.label[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self) -> int:
        return len(self.data)
    
    def load_data(self, data_dir: str) -> None: 
        pass
    