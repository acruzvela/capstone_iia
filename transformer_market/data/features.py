# transformer_market/data/features.py

'''
Objetivo de features.py
    Crear una función tipo:
    def build_market_features(df: pd.DataFrame) -> pd.DataFrame:
    que recibe el DataFrame crudo (OHLCV) y devuelve otro DataFrame con:
    date
    OHLCV (opcional mantenerlos)
    todas las features finales que acordamos

OHLCV es un formato estándar para representar datos de mercado financiero, especialmente 
de velas japonesas (candlesticks).
    La sigla significa:
    Letra	Significado	    Descripción
    O	    Open	        Precio de apertura del periodo.
    H	    High	        Precio máximo alcanzado en el periodo.
    L	    Low	            Precio mínimo alcanzado en el periodo.
    C	    Close	        Precio de cierre del periodo.
    V	    Volume	        Volumen negociado en el periodo (número de acciones, 
                            contratos, etc.).
'''


from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def build_market_features(
    df: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Calcula features técnicas a partir de OHLCV del IBEX.
    
    Requiere columnas: date, open, high, low, close, volume
    
    Features calculadas:
        - log_ret_1d, log_ret_5d, log_ret_20d: retornos logarítmicos
        - sma_ratio_10, sma_ratio_20: ratio close / SMA - 1
        - vol_10, vol_20: volatilidad móvil (std de log_ret_1d)
        - hl_range: (high - low) / close (volatilidad intradía)
        - oc_ret: ln(close / open) (fuerza neta del día)
        - upper_wick, lower_wick: mechas de velas normalizadas
        - range_vol_10: std del ln(high/low) en 10 días (Parkinson-like)
        - vol_z_20: z-score del volumen en 20 días
    
    ⚠️ Nota: Features técnicas tienen NaN esperados en primeras filas
    (rolling windows, SMA, z-score). Usar dropna() en dataset.py.
    
    Args:
        df: DataFrame con OHLCV (debe tener columnas: date, open, high, low, close, volume)
        output_dir: directorio para guardar CSV intermediate (optional)
    
    Returns:
        pd.DataFrame con features técnicas añadidas
        
    Raises:
        ValueError: si df está vacío o faltan columnas requeridas
    """
    # Validaciones de entrada
    if df.empty:
        raise ValueError("DataFrame está vacío")
    
    cols_required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in cols_required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    
    out = df.copy().reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    
    # Asegurar orden por date
    out = out.sort_values("date").reset_index(drop=True)
    
    logger.info(f"Calculando features técnicas: {len(out)} filas")

    # ===== Retornos log =====
    out["log_ret_1d"] = np.log(out["close"] / out["close"].shift(1))
    out["log_ret_5d"] = np.log(out["close"] / out["close"].shift(5))
    out["log_ret_20d"] = np.log(out["close"] / out["close"].shift(20))

    # ===== Tendencia (SMA ratios) =====
    sma10 = out["close"].rolling(10, min_periods=1).mean()
    sma20 = out["close"].rolling(20, min_periods=1).mean()
    
    # Guard: evitar división por cero
    out["sma_ratio_10"] = np.where(
        sma10 > 0,
        out["close"] / sma10 - 1.0,
        np.nan
    )
    out["sma_ratio_20"] = np.where(
        sma20 > 0,
        out["close"] / sma20 - 1.0,
        np.nan
    )

    # ===== Volatilidad (std rolling de log_ret_1d) =====
    out["vol_10"] = out["log_ret_1d"].rolling(10, min_periods=1).std()
    out["vol_20"] = out["log_ret_1d"].rolling(20, min_periods=1).std()

    '''
hl_range	Volatilidad intradía
    Alto → día “nervioso”, mucha pelea compradores/vendedores
    Bajo → día tranquilo

oc_ret	Fuerza neta del día
    Positivo → compradores dominaron el día
    Negativo → vendedores dominaron

upper_wick	Rechazo en máximos
    Grande → presión vendedora en la parte alta
    Pequeña → cierre cerca del máximo (fuerza)

lower_wick	Rechazo en mínimos
    Grande → fuerte entrada de compradores
    Pequeña → debilidad

range_vol_10	Inestabilidad reciente
    Alto → mercado inestable
    Bajo → mercado comprimido

vol_z_20	Volumen anómalo
    0 → volumen normal
    +2 → volumen excepcionalmente alto
    -2 → muy bajo

    '''
    # ===== Rango intradía =====
    out["hl_range"] = np.where(
        out["close"] > 0,
        (out["high"] - out["low"]) / out["close"],
        np.nan
    )

    # ===== Open->Close return (log) =====
    out["oc_ret"] = np.log(out["close"] / out["open"])

    # ===== Wicks normalizados =====
    # upper: high - max(open, close)
    oc_max = out[["open", "close"]].max(axis=1)
    oc_min = out[["open", "close"]].min(axis=1)

    out["upper_wick"] = np.where(
        out["close"] > 0,
        (out["high"] - oc_max) / out["close"],
        np.nan
    )
    out["lower_wick"] = np.where(
        out["close"] > 0,
        (oc_min - out["low"]) / out["close"],
        np.nan
    )

    # ===== Volatilidad del rango (Parkinson-like simple) =====
    # rango_log = ln(high/low)
    range_log = np.log(out["high"] / out["low"])
    out["range_vol_10"] = range_log.rolling(10, min_periods=1).std()

    # ===== Volumen normalizado (z-score 20) =====
    v = out["volume"].astype(float)
    v_mean20 = v.rolling(20, min_periods=1).mean()
    v_std20 = v.rolling(20, min_periods=1).std()
    out["vol_z_20"] = np.where(
        v_std20 > 1e-6,
        (v - v_mean20) / v_std20,
        np.nan
    )
    
    # ===== Avisos de NaN =====
    feature_cols = [
        "log_ret_1d", "log_ret_5d", "log_ret_20d",
        "sma_ratio_10", "sma_ratio_20",
        "vol_10", "vol_20",
        "hl_range", "oc_ret", "upper_wick", "lower_wick",
        "range_vol_10", "vol_z_20"
    ]
    
    for col in feature_cols:
        n_nan = out[col].isna().sum()
        if n_nan > 0:
            pct = 100 * n_nan / len(out)
            logger.debug(f"Feature '{col}': {n_nan} NaN ({pct:.1f}%)")
    
    logger.info(f"Features calculadas: {len(out)} filas, {len(feature_cols)} features")
    
    # ===== Guardar intermediate (optional) =====
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        features_path = output_dir / "features_intermediate.csv"
        out.to_csv(features_path, index=False)
        logger.info(f"Features intermediate guardadas en: {features_path}")
    
    return out

