"""Helpers shared by the entry points: seeding, device choice, honest timing."""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator a run uses.

    Called first in every entry point, so that the same config gives the same
    initial weights, the same dropout masks and the same samples.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN picks its algorithms by trial unless told not to, and some of them
    # are not deterministic.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(requested: str) -> torch.device:
    """Return the requested device, or the best available one if it is missing.

    The configs ask for "cuda". On a machine without it this falls back to
    Apple's "mps" and then to "cpu", and says so, because the device has to be
    reported next to any timing.
    """
    available = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "cpu": True,
    }
    if available.get(requested, False):
        return torch.device(requested)
    fallback = "mps" if available["mps"] else "cpu"
    print(f"device {requested!r} is not available, using {fallback!r}")
    return torch.device(fallback)


def synchronize(device: torch.device) -> None:
    """Wait until the device has finished the work queued on it.

    A GPU call returns before the computation is done. Reading the clock
    without waiting would time the queueing, not the work.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
