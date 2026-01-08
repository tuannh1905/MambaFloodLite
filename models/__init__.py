import os
import importlib

def get_model(model_name, num_classes=1):
    model_file = os.path.join(os.path.dirname(__file__), f'{model_name}.py')
    if not os.path.exists(model_file):
        raise ValueError(f"Model {model_name} not found")
    
    module = importlib.import_module(f'segmentation.models.{model_name}')
    return module.build_model(num_classes)