import torch
from omegaconf import DictConfig

from .CNN import CNN
from .ISANet import ISANet
from .MSAFCN import MSAFCN
from .TICNN import TICNN
from .WDCNN import WDCNN

def construct_model(model_cfg: DictConfig) -> torch.nn.Module:
    """
    Select model by name
    Args:
        model_cfg (dict): Configuration for the model
    return:
        model: model object
    """
    if "_name_" not in model_cfg:
        raise ValueError("Model name not found in config")
    
    model_name = model_cfg.get("_name_")
    model_cls = {
        'CNN': CNN,
        'ISANet': ISANet,
        'MSAFCN': MSAFCN,
        'TICNN': TICNN,
        'WDCNN': WDCNN,
    }[model_name]
    
    if model_name not in model_cls:
        raise ValueError(f"Model {model_name} not supported")
    model = model_cls(**model_cfg)
    return model

def model_identifier(model_cfg: DictConfig) -> str:
    """
    Get the model identifier
    Args:
        model_config (dict): Configuration for the model
    return:
        model_identifier: model identifier
    """
    model_name = model_cfg.get("_name_")
    model_cls = {
        'CNN': CNN,
        'ISANet': ISANet,
        'MSAFCN': MSAFCN,
        'TICNN': TICNN,
        'WDCNN': WDCNN,
    }[model_name]
    
    return model_cls.name(model_cfg)
