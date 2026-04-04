import os
import torch
import importlib
import inspect


def get_model(model_name, num_classes=1, seed=None, input_size=256):
    """
    Initialize model with DETERMINISTIC weights.

    Args:
        model_name: Name of model architecture
        num_classes: Number of output classes
        seed:        Random seed (REQUIRED for reproducibility)
        input_size:  Input image size — phải chia hết cho 16
                     Truyền vào build_model nếu model hỗ trợ tham số này.
    """
    # ✓ Validate input_size sớm, trước khi load bất cứ thứ gì
    if input_size % 16 != 0:
        raise ValueError(
            f"input_size={input_size} phải chia hết cho 16 "
            f"(do 4 lớp MaxPool2d×2 trong encoder)."
        )

    model_file = os.path.join(os.path.dirname(__file__), f'{model_name}.py')
    if not os.path.exists(model_file):
        available = [
            f.replace('.py', '')
            for f in os.listdir(os.path.dirname(__file__))
            if f.endswith('.py') and f != '__init__.py'
        ]
        raise ValueError(
            f"Model '{model_name}' not found.\n"
            f"Available: {', '.join(sorted(available))}"
        )

    # ✓ Set seed TRƯỚC khi khởi tạo model để weight init deterministic
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    module = importlib.import_module(f'models.{model_name}')

    # ✓ Dùng inspect để kiểm tra build_model có nhận input_size không
    #   → Tương thích ngược với các model cũ không có tham số này
    sig = inspect.signature(module.build_model)
    if 'input_size' in sig.parameters:
        model = module.build_model(num_classes=num_classes, input_size=input_size)
    else:
        model = module.build_model(num_classes=num_classes)

    if seed is not None:
        print(f"✓ Model '{model_name}' initialized with seed {seed}")

    return model