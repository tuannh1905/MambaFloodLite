from .bce import build_loss as build_bce
from .dice import build_loss as build_dice
from .ce import build_loss as build_ce
from .tversky import build_loss as build_tversky
import inspect

def get_loss(loss_name, num_classes=1):
    all_losses = {
        'bce': build_bce,
        'dice': build_dice,
        'ce': build_ce,
        'tversky': build_tversky
    }
    
    if loss_name not in all_losses:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(all_losses.keys())}")
    
    if num_classes == 1:
        if loss_name == 'ce':
            print("⚠️  Warning: CE loss với binary - recommend BCE/Dice")
        print(f"✓ Using {loss_name.upper()} for BINARY")
    else:
        if loss_name == 'bce':
            print("⚠️  Warning: BCE loss với multi-class - recommend CE/Dice")
        print(f"✓ Using {loss_name.upper()} for {num_classes}-class")
    
    # Check if build_loss accepts num_classes
    build_fn = all_losses[loss_name]
    sig = inspect.signature(build_fn)
    
    if 'num_classes' in sig.parameters:
        return build_fn(num_classes=num_classes)
    else:
        # Backward compatibility: loss cũ không có num_classes param
        print(f"⚠️  {loss_name} không hỗ trợ num_classes, dùng default")
        return build_fn()