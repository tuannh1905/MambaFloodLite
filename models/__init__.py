import os
import torch
import importlib

def get_model(model_name, num_classes=1, seed=None):
    """
    Initialize model with deterministic weights
    
    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        seed: Random seed for weight initialization (CRITICAL for reproducibility)
    """
    model_file = os.path.join(os.path.dirname(__file__), f'{model_name}.py')
    if not os.path.exists(model_file):
        raise ValueError(f"Model {model_name} not found")
    
    # CRITICAL: Set seed before model initialization
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    module = importlib.import_module(f'models.{model_name}')
    return module.build_model(num_classes)