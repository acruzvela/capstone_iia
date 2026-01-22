# transformer_market/data/load_sqlite.py

'''
Responsabilidad exacta de load_sqlite.py
    Conectarse a la base de datos (ibex35.db)
    Leer la tabla ibex_index
    Convertir date a datetime
    Ordenar por fecha
    Validar:
        que no haya duplicados
        que no falten columnas
    Tratar el volumen:
        volume == 0 → NaN
    Devolver un pd.DataFrame
'''


from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}


def load_ibex_index(db_path: Path, table: str = "ibex_index") -> pd.DataFrame:
    """
    Carga la tabla ibex_index desde SQLite y devuelve un DataFrame limpio y ordenado.

    Responsabilidad:
      - Leer datos crudos
      - Convertir date a datetime
      - Ordenar por fecha
      - Chequeos mínimos
      - volume == 0 -> NaN (faltante/no reportado)

    NO hace:
      - Features
      - Targets
      - Escalado
      - Ventanas
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"No existe la base de datos: {db_path}")

    query = f"""
        SELECT date, open, high, low, close, volume
        FROM {table}
        ORDER BY date ASC
    """

    with sqlite3.connect(str(db_path)) as conn:
        df = pd.read_sql_query(query, conn)

    # Validación de columnas
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en {table}: {sorted(missing)}")

    if df.empty:
        raise ValueError(f"La tabla {table} está vacía.")

    # Convertir date a datetime (tu tabla guarda texto ISO, perfecto)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)

    if df["date"].isna().any():
        bad = df.loc[df["date"].isna(), "date"]
        raise ValueError(f"Fechas inválidas detectadas al parsear date en {table}.")

    # Orden y duplicados defensivos (aunque sea PK)
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].head(5).tolist()
        raise ValueError(f"Fechas duplicadas detectadas (muestra): {dups}")

    # Chequeo de coherencia OHLC (defensivo, no bloquea por valores raros pequeños)
    # Si prefieres, podemos convertirlo a warning en vez de raise.
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Inconsistencia: hay filas donde high < max(open, close, low).")
    if (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Inconsistencia: hay filas donde low > min(open, close, high).")

    # Volume: en índices a veces viene 0 en los más recientes -> tratamos como faltante
    # (importante para vol_z_20 y para que el modelo no interprete 0 como señal)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df.loc[df["volume"] == 0, "volume"] = pd.NA

    return df



