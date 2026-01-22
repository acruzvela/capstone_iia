# transformer_market/data/splits.py

'''
Split temporal 70 / 15 / 15 (por orden, no aleatorio)

Responsabilidad
    Recibir n_samples
    Devolver índices de train / val / test
    Nada de tocar datos

'''


from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


def temporal_split_indices(
    n_samples: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict[str, np.ndarray]:
    """
    Genera índices temporales para train/val/test sin mezcla.
    
    Usa split por orden cronológico (no aleatorio):
        - Train: primeras n_train muestras
        - Val: siguientes n_val muestras
        - Test: muestras restantes
    
    Args:
        n_samples: número total de muestras
        train_ratio: proporción para train (default: 0.70)
        val_ratio: proporción para validación (default: 0.15)
                   test_ratio = 1 - train_ratio - val_ratio (default: 0.15)
    
    Returns:
        dict con claves "train", "val", "test"
        → np.ndarray índices (preservando orden cronológico)
        
    Raises:
        ValueError: si ratios inválidos o sum >= 1.0
    """
    # Validación de entrada
    if n_samples <= 0:
        raise ValueError(f"n_samples debe ser > 0 (recibido: {n_samples})")
    
    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio debe estar en (0, 1) (recibido: {train_ratio})")
    
    if not (0 < val_ratio < 1):
        raise ValueError(f"val_ratio debe estar en (0, 1) (recibido: {val_ratio})")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError(f"train_ratio + val_ratio debe ser < 1 (recibido: {train_ratio + val_ratio})")

    test_ratio = 1.0 - train_ratio - val_ratio
    
    logger.info(f"Split temporal: train={train_ratio*100:.0f}%, val={val_ratio*100:.0f}%, test={test_ratio*100:.0f}%")

    idx = np.arange(n_samples)

    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    # Validación: splits no vacíos
    if len(train_idx) == 0:
        raise ValueError(f"train split vacío (n_samples={n_samples}, train_ratio={train_ratio})")
    if len(val_idx) == 0:
        raise ValueError(f"val split vacío (n_samples={n_samples}, val_ratio={val_ratio})")
    if len(test_idx) == 0:
        raise ValueError(f"test split vacío (n_samples={n_samples})")
    
    # Reporte de splits
    logger.debug(f"Indices: train={len(train_idx)} ({train_idx[0]}:{train_idx[-1]}), val={len(val_idx)} ({val_idx[0]}:{val_idx[-1]}), test={len(test_idx)} ({test_idx[0]}:{test_idx[-1]})")
    logger.info(f"Splits generados: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)} (total={len(train_idx)+len(val_idx)+len(test_idx)})")

    return {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }
