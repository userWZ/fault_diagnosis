import torch
from .cwru import CWRU
from .hgd import HGD
from .pu import PU
from torch.utils.data import Dataset, DataLoader

def get_dataset(dataset_cfg) -> Dataset:
    """
    get dataset by type
    Args:
        dataset_cfg (dict): Configuration for the dataset
    return:
        dataset: dataset object
    """
    if "_name_" not in dataset_cfg:
        raise ValueError("Dataset name not found in config")
    
    dataset_name = dataset_cfg["_name_"]
    type2data = {
        'CWRU': CWRU(**dataset_cfg),
        'HGD': HGD(**dataset_cfg),
        'PU': PU(**dataset_cfg),
    }
    
    if dataset_name not in type2data:
        raise ValueError(f"Dataset {dataset_name} not supported")
    dataset = type2data[dataset_name]
    return dataset

def get_dataloader(dataset_cfg) -> tuple[DataLoader]:
    """
    get dataloader by type
    Args:
        dataset_cfg (dict): Configuration for the dataset
        is_train (bool): Whether to use the training set
    return:
        dataloader: dataloader object
    """
    dataset = get_dataset(dataset_cfg)
    
    # split train set and test set
    test_size = int(dataset_cfg["test_size"] * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # create dataloader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, test_loader

    