import os
import torch
import importlib
import inspect
def get_model(model_name, num_classes=1, seed=None, input_size = 256):
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
    if input_size % 16 != 0:
        raise ValueError(f"lỗi input_size ({input_size}) phải chia hết cho 16")
    print(f"\n{'='*50}")
    if input_size <= 128:
        print(f" [MCU ULTRA-LIGHT] Chế độ siêu nhẹ: Kích thước {input_size}x{input_size}.")
    elif input_size == 256:
        print(f" [STANDARD] Chế độ tiêu chuẩn: Kích thước 256x256.")
    else:
        print(f"🚀 [HIGH-RES] Chế độ độ phân giải cao: Kích thước {input_size}x{input_size}.")
    print(f"{'='*50}\n")

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
    
    # Dùng Inspect để xem model có hỗ trợ cấu hình Static FCN (nhận input_size) không
    sig = inspect.signature(module.build_model)
    
    if 'input_size' in sig.parameters:
        # NẾU CÓ: Bơm input_size vào để nó cắt giảm phép tính tĩnh (Cho LiteV8)
        model = module.build_model(num_classes=num_classes, input_size=input_size)
    else:
        # NẾU KHÔNG: Fallback an toàn cho các mô hình cũ
        model = module.build_model(num_classes=num_classes)
    # -------------------------------------------
    
    if seed is not None:
        print(f"✓ Model '{model_name}' initialized with seed {seed}")
        
    return model