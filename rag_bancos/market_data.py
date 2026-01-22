import logging
from pathlib import Path
import sqlite3
import pandas as pd

logger = logging.getLogger(__name__)

def load_stock_market_prices(db_path: str | Path,
                             tickers: list[str] | None = None,
                             date_from: str | None = None,
                             date_to: str | None = None) -> pd.DataFrame:
    """
    Lee precios (date,ticker,close,volume,...) desde ibex35.db (tabla stock_market).
    Devuelve DF con date parseado a datetime64[ns] sin timezone.
    
    Args:
        db_path: ruta a base de datos SQLite
        tickers: lista de tickers a filtrar (None = todos)
        date_from: fecha mínima (formato YYYY-MM-DD)
        date_to: fecha máxima (formato YYYY-MM-DD)
    
    Returns:
        pd.DataFrame con columnas: date, ticker, open, high, low, close, volume
    """
    db_path = str(db_path)
    where = []
    params = []

    if tickers:
        where.append(f"ticker IN ({','.join(['?']*len(tickers))})")
        params.extend(tickers)

    if date_from:
        where.append("date >= ?")
        params.append(date_from)

    if date_to:
        where.append("date <= ?")
        params.append(date_to)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    q = f"""
        SELECT date, ticker, open, high, low, close, volume
        FROM stock_market
        {where_sql}
        ORDER BY ticker, date
    """

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(q, conn, params=params, parse_dates=["date"])
    finally:
        conn.close()

    logger.info("Cargados %d registros de precios (tickers=%d)", len(df), len(df['ticker'].unique()) if len(df) > 0 else 0)
    return df


def build_ibex_proxy_daily(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Construye un proxy diario del IBEX a partir de retornos diarios cross-section.
    
    Nota: El primer día de cada ticker tiene NaN en ret_d (pct_change).
    Se descarta en la agregación por fecha.
    
    Args:
        df_prices: DF con columnas: date, ticker, close
    
    Returns:
        pd.DataFrame con columnas: date, ibex_ret_d, n_stocks
        - ibex_ret_d: media de retornos diarios cross-section
        - n_stocks: número de acciones con retorno válido ese día
    """
    d = df_prices.copy()
    d = d.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    
    if d.empty:
        logger.warning("build_ibex_proxy_daily: df_prices vacío")
        return pd.DataFrame(columns=["date", "ibex_ret_d", "n_stocks"])
    
    d["ret_d"] = d.groupby("ticker")["close"].pct_change()
    
    ibex = (
        d.dropna(subset=["ret_d"])
        .groupby("date", as_index=False)
         .agg(
             ibex_ret_d=("ret_d", "mean"),
             n_stocks=("ret_d", "count"),
         )
         .sort_values("date")
    )
    
    # aviso si hay fechas con pocos stocks
    low_count = ibex[ibex["n_stocks"] < 3]
    if not low_count.empty:
        logger.warning("build_ibex_proxy_daily: %d fechas con <3 acciones validas", len(low_count))
    
    return ibex





