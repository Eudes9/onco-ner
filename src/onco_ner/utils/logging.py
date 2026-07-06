# src/onco_ner/utils/logging.py

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    
    # Évite de ré-ajouter des handlers si le logger a déjà été configuré
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        # Empêche la duplication des logs (Hydra, FastAPI)
        logger.propagate = False
        
    return logger