# transformer_market/data/scaling.py

'''
Escalado SOLO con train + imputación neutral

Decisiones (ya cerradas)
    Scaler: StandardScaler
    Fit: solo train
    NaNs:
        se mantienen hasta escalar
        se imputan a 0 después del escalado

Guardamos:
    .npz escalado
    scaler.pkl
    scaling_summary.json

'''



from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def scale_and_impute_npz(
    npz_path: Path | str,
    output_dir: Path | str,
    split_indices: dict[str, np.ndarray],
) -> dict[str, Path]:
    """
    Escala X usando SOLO train, imputa NaNs a 0 tras escalado.
    
    Workflow:
        1. Carga NPZ (X, y_reg, y_cls, dates, feature_cols)
        2. Ajusta StandardScaler solo con train
        3. Transforma todo con scaler
        4. Imputa NaNs → 0 (después de transformación)
        5. Guarda: NPZ escalado, scaler.pkl, scaling_summary.json
    
    Args:
        npz_path: Path o str a ibex_w60_p5.npz
        output_dir: Path o str para guardar artefactos (dataset/)
        split_indices: dict con claves "train", "val", "test" 
                      → np.ndarray índices. Ej: {"train": [0,1,2,...], ...}
    
    Returns:
        dict con claves: "scaled_npz", "scaler", "summary"
        → Paths a los archivos generados
        
    Raises:
        FileNotFoundError: si npz_path no existe
        ValueError: si split_indices inválido, NPZ corrupto, o shapes inconsistentes
        KeyError: si faltan claves en split_indices o en NPZ
    """
    # Validaciones de entrada
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ no encontrado: {npz_path}")
    
    output_dir = Path(output_dir)
    
    logger.info(f"Cargando NPZ: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]          # (N, W, F)
        y_reg = data["y_reg"]
        y_cls = data["y_cls"]
        dates = data["dates"]
        feature_cols = data["feature_cols"]
    except KeyError as e:
        raise KeyError(f"NPZ corrupto o falta clave {e}. Esperadas: X, y_reg, y_cls, dates, feature_cols") from e
    except Exception as e:
        raise ValueError(f"Error cargando NPZ: {e}") from e
    
    N, W, F = X.shape
    logger.info(f"NPZ cargado: X.shape={X.shape}, features={list(feature_cols)}")

    # Validar split_indices
    required_splits = {"train", "val", "test"}
    missing_splits = required_splits - set(split_indices.keys())
    if missing_splits:
        raise ValueError(f"split_indices falta claves: {missing_splits}")
    
    for split_name, indices in split_indices.items():
        if len(indices) == 0:
            raise ValueError(f"split_indices['{split_name}'] está vacío")
        if indices.max() >= N or indices.min() < 0:
            raise ValueError(f"split_indices['{split_name}'] fuera de rango [0, {N})")
    
    # Chequeo de solapamiento
    all_indices = np.concatenate([split_indices["train"], split_indices["val"], split_indices["test"]])
    if len(all_indices) != len(np.unique(all_indices)):
        logger.warning("⚠️ split_indices con solapamientos detectados")
    
    logger.debug(f"Splits: train={len(split_indices['train'])}, val={len(split_indices['val'])}, test={len(split_indices['test'])}")
    
    # ===== Ajuste del scaler SOLO con train =====
    train_idx = split_indices["train"]

    # Aplanamos para escalar por feature
    X_train_flat = X[train_idx].reshape(-1, F)
    
    logger.info(f"Ajustando StandardScaler con {len(train_idx)} muestras de train ({len(train_idx)*W} puntos)")
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    # Reporte del scaler
    mean_str = ", ".join([f"{m:.4f}" for m in scaler.mean_[:3]] + ["..."])
    std_str = ", ".join([f"{s:.4f}" for s in scaler.scale_[:3]] + ["..."])
    logger.debug(f"Scaler mean=[{mean_str}], std=[{std_str}]")

    # ===== Transformación =====
    logger.info("Transformando dataset completo...")
    X_scaled = scaler.transform(X.reshape(-1, F)).reshape(N, W, F)
    
    # Chequeo de NaNs antes de imputación
    n_nan_before = np.isnan(X_scaled).sum()
    if n_nan_before > 0:
        pct_nan = 100 * n_nan_before / X_scaled.size
        logger.warning(f"NaNs detectados tras transformación: {n_nan_before} ({pct_nan:.2f}%)")

    # ===== Imputación neutral =====
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    n_nan_after = np.isnan(X_scaled).sum()
    
    if n_nan_after > 0:
        logger.error(f"Error: {n_nan_after} NaNs persisten después de imputación")
        raise ValueError(f"Imputación falló: {n_nan_after} NaNs en X_scaled")
    
    logger.info(f"Imputación completada: {n_nan_before} NaNs → 0")

    # ===== Guardar NPZ escalado =====
    scaled_dir = output_dir / "dataset"
    scaled_dir.mkdir(parents=True, exist_ok=True)
    
    scaled_npz_path = scaled_dir / "ibex_w60_p5_scaled.npz"
    np.savez_compressed(
        scaled_npz_path,
        X=X_scaled,
        y_reg=y_reg,
        y_cls=y_cls,
        dates=dates,
        feature_cols=feature_cols,
        split_indices=split_indices,
    )
    file_size_mb = scaled_npz_path.stat().st_size / (1024 * 1024)
    logger.info(f"NPZ escalado guardado: {scaled_npz_path} ({file_size_mb:.2f} MB)")

    # ===== Guardar scaler =====
    scaler_path = scaled_dir / "scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler guardado: {scaler_path}")

    # ===== Guardar resumen =====
    summary = {
        "n_samples": int(N),
        "window": int(W),
        "n_features": int(F),
        "train_size": int(len(split_indices["train"])),
        "val_size": int(len(split_indices["val"])),
        "test_size": int(len(split_indices["test"])),
        "features": feature_cols.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    summary_path = scaled_dir / "scaling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary guardado: {summary_path}")
    
    logger.info(f"Escalado completado: X_scaled.shape={X_scaled.shape}, X_scaled.min={X_scaled.min():.4f}, X_scaled.max={X_scaled.max():.4f}")

    return {
        "scaled_npz": scaled_npz_path,
        "scaler": scaler_path,
        "summary": summary_path,
    }


