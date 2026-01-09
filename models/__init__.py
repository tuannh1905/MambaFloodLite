import os
import torch
import importlib

def get_model(model_name, num_classes=1, seed=None):
    """
    Initialize model with DETERMINISTIC weights
    
    CRITICAL for reproducibility:
    - Sets seed before model creation
    - Ensures consistent weight initialization
    - Model architecture must not use random operations
    
    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        seed: Random seed (REQUIRED for reproducibility)
    
    Returns:
        Initialized model with deterministic weights
    """
    
    model_file = os.path.join(os.path.dirname(__file__), f'{model_name}.py')
    if not os.path.exists(model_file):
        available_models = [
            f.replace('.py', '') for f in os.listdir(os.path.dirname(__file__))
            if f.endswith('.py') and f != '__init__.py'
        ]
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available models: {', '.join(available_models)}"
        )
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    module = importlib.import_module(f'models.{model_name}')
    model = module.build_model(num_classes)
    
    if seed is not None:
        print(f"✓ Model '{model_name}' initialized with seed {seed}")
    
    return model