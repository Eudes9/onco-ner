# src/onco_ner/utils/seed.py

import random

import numpy as np
import torch

from onco_ner.utils.logging import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Fixe tous les seeds pour la reproductibilité."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Seed fixé à {seed}")