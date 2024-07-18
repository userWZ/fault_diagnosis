import torch

def to_tensor(sample):
    features, label = sample
    return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)