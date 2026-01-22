# transformer_market/data/dataset.py

'''
Responsabilidad exacta de dataset.py
    Recibir el DataFrame con features ya construidas
    Crear los targets P5:
        y_reg_5d
        y_cls_5d
    Seleccionar solo las columnas feature finales
    Eliminar filas no utilizables (por NaNs estructurales)
    Construir ventanas W60
    Guardar:
        X, y_reg, y_cls, dates → .npz
    un dataset_summary.csv (para UI/debug)
'''

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "log_ret_1d",
    "log_ret_5d",
    "log_ret_20d",
    "sma_ratio_10",
    "sma_ratio_20",
    "vol_10",
    "vol_20",
    "hl_range",
    "oc_ret",
    "upper_wick",
    "lower_wick",
    "range_vol_10",
    "vol_z_20",
]


def build_dataset_w60_p5(
    df: pd.DataFrame,
    output_dir: Path,
    window: int = 60,
    horizon: int = 5,
) -> Dict[str, Path | int]:
    """
    Construye dataset para Transformer:
      - Target P5 (regresión + clasificación)
      - Ventanas W60
      - Guarda NPZ + CSV resumen

    Args:
        df: DataFrame con features y columna 'close'
        output_dir: directorio de salida
        window: tamaño de ventana temporal (default: 60)
        horizon: días futuros para target (default: 5)

    Returns:
        Dict con claves: "npz", "summary", "n_samples", "window", "n_features"
        
    Raises:
        ValueError: si df está vacío, faltan columnas o window/horizon inválidos
    """
    # Validaciones de entrada
    if df.empty:
        raise ValueError("DataFrame está vacío")
    
    if window <= 0 or horizon <= 0:
        raise ValueError(f"window y horizon deben ser > 0 (window={window}, horizon={horizon})")
    
    cols_needed = ["date", "close"] + FEATURE_COLS
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    
    output_dir = Path(output_dir)
    dataset_dir = output_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    data = df.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    
    logger.info(f"Dataset inicial: {len(data)} filas, fecha rango [{data['date'].min()}, {data['date'].max()}]")

    # ===== Target P5 =====
    data["y_reg_5d"] = np.log(data["close"].shift(-horizon) / data["close"])
    data["y_cls_5d"] = (data["y_reg_5d"] > 0).astype(int)

    # ===== Selección de columnas necesarias =====
    cols_needed = ["date", "close", "y_reg_5d", "y_cls_5d"] + FEATURE_COLS
    data = data[cols_needed]

    # ===== Eliminar filas con NaNs estructurales =====
    # vol_z_20 puede ser NaN si hay pocos datos; lo permitimos
    FEATURE_COLS_CORE = [c for c in FEATURE_COLS if c != "vol_z_20"]
    
    n_before = len(data)
    data = data.dropna(subset=FEATURE_COLS_CORE + ["y_reg_5d"]).reset_index(drop=True)
    n_after = len(data)
    n_dropped = n_before - n_after
    
    if n_dropped > 0:
        pct = 100 * n_dropped / n_before
        logger.warning(f"Descartadas {n_dropped} filas ({pct:.1f}%) por NaN")
    
    if n_after == 0:
        logger.error("Dataset vacío después de eliminar NaNs")
        raise ValueError("No hay datos válidos después de eliminar NaNs")


    # ===== Construcción de ventanas =====
    X_list, y_reg_list, y_cls_list, dates = [], [], [], []

    for i in range(window - 1, len(data)):
        X_window = data.loc[i - window + 1 : i, FEATURE_COLS].values
        y_reg = data.loc[i, "y_reg_5d"]
        y_cls = data.loc[i, "y_cls_5d"]
        date = data.loc[i, "date"]

        # Validación: ventana debe tener shape correcto
        if X_window.shape[0] != window or X_window.shape[1] != len(FEATURE_COLS):
            logger.debug(f"Saltando muestra i={i}: shape inválido {X_window.shape}")
            continue

        X_list.append(X_window)
        y_reg_list.append(y_reg)
        y_cls_list.append(y_cls)
        dates.append(date)

    if not X_list:
        logger.error(f"No se construyeron ventanas (window={window}, datos={len(data)})")
        raise ValueError("No hay ventanas válidas después del procesamiento")

    X = np.stack(X_list)
    y_reg = np.array(y_reg_list)
    y_cls = np.array(y_cls_list)
    dates = np.array(dates)

    logger.info(f"Dataset construido: X.shape={X.shape}, y_reg.shape={y_reg.shape}")

    # ===== Guardar NPZ =====
    npz_path = dataset_dir / "ibex_w60_p5.npz"
    np.savez_compressed(
        npz_path,
        X=X,
        y_reg=y_reg,
        y_cls=y_cls,
        dates=dates,
        feature_cols=np.array(FEATURE_COLS),
        window=window,
        horizon=horizon,
    )
    file_size_mb = npz_path.stat().st_size / (1024 * 1024)
    logger.info(f"NPZ guardado en: {npz_path} ({file_size_mb:.2f} MB)")

    # ===== Dataset summary (para UI / inspección) =====
    summary = pd.DataFrame({
        "date": dates,
        "y_reg_5d": y_reg,
        "y_cls_5d": y_cls,
    })

    summary_path = dataset_dir / "dataset_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info(f"Summary guardado en: {summary_path}")

    return {
        "npz": npz_path,
        "summary": summary_path,
        "n_samples": X.shape[0],
        "window": window,
        "n_features": X.shape[2],
    }
