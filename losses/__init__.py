from .bce          import build_loss as build_bce
from .dice         import build_loss as build_dice
from .focal        import build_loss as build_focal
from .tversky      import build_loss as build_tversky
from .lovasz       import build_loss as build_lovasz
from .lovasz_focal import build_loss as build_lovasz_focal
import inspect

def get_loss(loss_name, num_classes=1):
    all_losses = {
        'bce':          build_bce,
        'dice':         build_dice,
        'focal':        build_focal,
        'tversky':      build_tversky,
        'lovasz':       build_lovasz,         # ← standalone Lovász-Hinge
        'lovasz_focal': build_lovasz_focal,   # ← Focal + Lovász hybrid
    }

    if loss_name not in all_losses:
        raise ValueError(
            f"Unknown loss: '{loss_name}'. "
            f"Available: {list(all_losses.keys())}"
        )

    tag = "BINARY" if num_classes == 1 else f"{num_classes}-class"
    print(f"✓ Using {loss_name.upper()} for {tag}")

    build_fn = all_losses[loss_name]
    if 'num_classes' in inspect.signature(build_fn).parameters:
        return build_fn(num_classes=num_classes)
    return build_fn()
