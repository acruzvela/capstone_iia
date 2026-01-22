# ibex_utils.py
import os
import sqlite3
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import requests
from dotenv import load_dotenv

from agente.config import DB_PATH

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------


load_dotenv()
EODHD_API_KEY = os.environ.get("EODHD_API_KEY")


def get_conn():
    return sqlite3.connect(DB_PATH)


def limpiar_fecha(fecha_sql: str) -> str:
    """
    Convierte '2025-12-05 00:00:00' -> '2025-12-05'
    """
    return fecha_sql.split(" ")[0]


def yf_ticker(ticker: str) -> str:
    """
    Convierte SAN -> SAN.MC
    """
    return f"{ticker}.MC"


def normalizar_df_yf(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que:
      - Date pasa a columna normal
      - Columnas no son MultiIndex
    """
    if df.index.name == "Date" or "Date" not in df.columns:
        df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def guardar_stock_append(df: pd.DataFrame) -> int:
    """
    Inserta en stock_market con if_exists='append'.
    Devuelve cuántas filas se han insertado.
    """
    if df is None or df.empty:
        return 0
    with get_conn() as conn:
        df.to_sql("stock_market", conn, if_exists="append", index=False)
    return int(len(df))


def obtener_ultima_fecha_db(ticker: str) -> str | None:
    """
    Devuelve la última fecha (YYYY-MM-DD) para un ticker o None si no hay datos.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM stock_market WHERE ticker = ?", (ticker,))
    val = cur.fetchone()[0]
    conn.close()
    if val is None:
        return None
    return limpiar_fecha(val)


# -------------------------------------------------------------------
# Tool 1: estado de la BBDD
# -------------------------------------------------------------------
def get_status(tickers: List[str]) -> Dict[str, Any]:
    """
    Devuelve para cada ticker:
    {
        "SAN": {"exists": True, "last_date": "2025-12-05"},
        "ACX": {"exists": False, "last_date": None}
    }
    """
    out: Dict[str, Any] = {}
    for t in tickers:
        T = t.upper()
        last = obtener_ultima_fecha_db(T)
        if last is None:
            out[T] = {"exists": False, "last_date": None}
        else:
            out[T] = {"exists": True, "last_date": last}
    return out


# -------------------------------------------------------------------
# Tool 2: carga completa (full load, sólo yfinance)
# -------------------------------------------------------------------
def script_full_load(tickers, from_date="2023-01-01"):
    to_date = datetime.today().strftime("%Y-%m-%d")

    resumen = {}
    dfs = []

    for t in tickers:
        T = t.upper()
        symbol = yf_ticker(T)

        print(f"[FULL] {symbol} {from_date} → {to_date}")

        df_yf = yf.download(
            symbol, start=from_date, end=to_date,
            interval="1d", auto_adjust=True
        )

        if df_yf.empty:
            resumen[T] = {
                "status": "full_load_no_data",
                "from": from_date,
                "to": None,
                "new_rows": 0,
                "used_fallback": False,
                "fallback_rows": 0
            }
            continue

        df_yf = normalizar_df_yf(df_yf)
        df_yf = df_yf.rename(columns={
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Adj Close": "adj_close",
            "Volume": "volume"
        })
        df_yf["ticker"] = T
        df_yf = df_yf[["date", "ticker", "open", "high", "low", "close", "volume"]]

        last_yf = df_yf["date"].max().strftime("%Y-%m-%d")

        df_gap, fallback_rows, used_fallback = None, 0, False
        if last_yf < to_date:
            print(f"[FULL] {T}: YF hasta {last_yf}, intentamos EODHD…")
            df_gap, fallback_rows, used_fallback = descargar_gap_eodhd(T, last_yf, to_date)

        if df_gap is not None:
            df_comb = pd.concat([df_yf, df_gap], ignore_index=True)
        else:
            df_comb = df_yf

        df_comb = df_comb.sort_values("date").drop_duplicates(subset=["date", "ticker"])
        dfs.append(df_comb)

        resumen[T] = {
            "status": "full_load",
            "new_rows": len(df_comb),
            "from": from_date,
            "to": df_comb["date"].max().strftime("%Y-%m-%d"),
            "used_fallback": used_fallback,
            "fallback_rows": fallback_rows
        }

    if dfs:
        guardar_stock_append(pd.concat(dfs, ignore_index=True))

    return resumen


# -------------------------------------------------------------------
# Auxiliar: descarga incremental para UN ticker (YF + fallback EODHD)
# -------------------------------------------------------------------
def descargar_ticker_incremental(ticker: str, last_date: str) -> pd.DataFrame | None:
    """
    Descarga datos a partir del día siguiente a last_date hasta hoy.
    Primero usa yfinance; si no hay datos, intenta EODHD (si hay API key).
    Devuelve un DataFrame listo para insertar o None si no hay datos.
    """
    dt_inicio = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    fecha_yf = dt_inicio.strftime("%Y-%m-%d")
    hoy = datetime.today().strftime("%Y-%m-%d")

    if fecha_yf > hoy:
        # Ya está al día
        print(f"[INC] {ticker}: ya está actualizado (last_date={last_date})")
        return None

    symbol = yf_ticker(ticker)
    print(f"[INC][YF] {symbol} desde {fecha_yf}")
    df = yf.download(symbol, start=fecha_yf)

    if df.empty:
        print(f"[INC][YF] {ticker}: sin datos nuevos, probando EODHD…")
        if not EODHD_API_KEY:
            print("[INC][EOD] No hay EODHD_API_KEY, no puedo hacer fallback.")
            return None

        url = (
            f"https://eodhd.com/api/eod/{symbol}"
            f"?from={fecha_yf}&to={hoy}&period=d"
            f"&api_token={EODHD_API_KEY}&fmt=json"
        )
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[INC][EOD] Error HTTP para {ticker}: {e}")
            return None

        if not data:
            print(f"[INC][EOD] {ticker}: sin datos nuevos tampoco en EODHD")
            return None

        df = pd.DataFrame(data)
        if "date" not in df.columns:
            print(f"[INC][EOD] {ticker}: respuesta sin 'date'")
            return None

        df["date"] = pd.to_datetime(df["date"])
        cols = ["date"]
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                cols.append(c)
        df = df[cols]
    else:
        df = normalizar_df_yf(df)
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

    df["ticker"] = ticker
    cols_final = ["date", "open", "high", "low", "close", "volume", "ticker"]
    cols_final = [c for c in cols_final if c in df.columns]
    df = df[cols_final]

    return df

def descargar_gap_eodhd(ticker: str, last_date: str, to_date: str):
    """
    Rellena el tramo final usando SOLO EODHD, entre last_date+1 y to_date.
    Solo se realiza si el hueco es <= 365 días.
    Devuelve (df, fallback_rows, used_fallback)
    """
    if not EODHD_API_KEY:
        print(f"[GAP][EOD] Sin API key: imposible rellenar gap de {ticker}")
        return None, 0, False

    dt_inicio = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    dt_fin = datetime.strptime(to_date, "%Y-%m-%d")

    if dt_inicio > dt_fin:
        return None, 0, False

    gap_days = (dt_fin - dt_inicio).days
    if gap_days > 365:
        print(f"[GAP][EOD] Gap {gap_days} días > 365: no permitido en cuenta free.")
        return None, 0, False

    from_str = dt_inicio.strftime("%Y-%m-%d")
    to_str = dt_fin.strftime("%Y-%m-%d")
    symbol = yf_ticker(ticker)

    print(f"[GAP][EOD] {ticker}: rellenando desde {from_str} hasta {to_str}")

    url = (
        f"https://eodhd.com/api/eod/{symbol}"
        f"?from={from_str}&to={to_str}&period=d"
        f"&api_token={EODHD_API_KEY}&fmt=json"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[GAP][EOD] Error para {ticker}: {e}")
        return None, 0, False

    if not data:
        print(f"[GAP][EOD] Sin datos EODHD para gap de {ticker}")
        return None, 0, False

    df = pd.DataFrame(data)
    if "date" not in df.columns:
        return None, 0, False

    df["date"] = pd.to_datetime(df["date"])
    cols = ["date"] + [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[cols]
    df["ticker"] = ticker

    return df, len(df), True

# -------------------------------------------------------------------
# Tool 3: actualización incremental
# -------------------------------------------------------------------
def script_incremental(tickers):
    resumen = {}

    for t in tickers:
        T = t.upper()
        last = obtener_ultima_fecha_db(T)

        if last is None:
            resumen[T] = {
                "status": "no_data_in_db",
                "new_rows": 0,
                "last_date": None,
                "used_fallback": False,
                "fallback_rows": 0
            }
            continue

        dt_last = datetime.strptime(last, "%Y-%m-%d")
        dt_today = datetime.today()
        gap_days = (dt_today - dt_last).days

        # Nuevo control del límite free de EODHD
        if gap_days > 365:
            print(f"[INC] {T}: gap {gap_days} > 365 → no se puede hacer incremental")
            resumen[T] = {
                "status": "too_old_for_incremental",
                "new_rows": 0,
                "last_date": last,
                "used_fallback": False,
                "fallback_rows": 0,
                "gap_days": gap_days
            }
            continue

        # Intento YF
        df = descargar_ticker_incremental(T, last)
        used_fallback = False
        fallback_rows = 0

        if df is None or df.empty:
            # Intentamos EODHD manualmente (seguro que gap <= 365)
            df, fallback_rows, used_fallback = descargar_gap_eodhd(T, last, dt_today.strftime("%Y-%m-%d"))
            if df is None or df.empty:
                resumen[T] = {
                    "status": "no_new_data",
                    "new_rows": 0,
                    "last_date": last,
                    "used_fallback": False,
                    "fallback_rows": 0
                }
                continue

        filas = guardar_stock_append(df)
        new_last = obtener_ultima_fecha_db(T)

        resumen[T] = {
            "status": "updated",
            "new_rows": filas,
            "last_date": new_last,
            "used_fallback": used_fallback,
            "fallback_rows": fallback_rows
        }

    return resumen



