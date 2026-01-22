import pandas as pd
import numpy as np
from datetime import date as _date

from .config import SOURCES

def agregar_diario(df_docs: pd.DataFrame, sources=SOURCES):
    """
    df_docs: filas por documento con columnas: fuente, date, score, pct_signal, n_trozos
    Devuelve df_daily con una fila por (fuente, date).
    Incluye BCE aunque no tenga datos (aparecerá con NaN/0 docs al reindexar).
    """
    # quitamos docs sin fecha (por si hay nombres raros)
    df = df_docs.dropna(subset=["date"]).copy()

    # 🔧 CLAVE: si date viene como datetime (con hora), lo pasamos a "día"
    # - si ya es date, no rompe nada
    #df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.date
    df["date"] = pd.to_datetime(df["date"]).dt.date

    
    today = _date.today()
    df = df[df["date"] <= today].copy()

    #agregación diaria
    agg = (
        df.groupby(["fuente", "date"])
        .agg(
            sent_mean=("score", "mean"),
            sent_median=("score", "median"),
            sent_std=("score", "std"),
            signal_mean=("pct_signal", "mean"),
            n_docs=("score", "size"),
        )
        .reset_index()
    )

    # calendario común (todas las fechas presentes en cualquier fuente)
    if len(agg) == 0:
        # caso extremo: no hay nada
        return agg

    all_dates = pd.Index(sorted(agg["date"].unique()), name="date")
    all_sources = pd.Index(sources, name="fuente")
    full_index = pd.MultiIndex.from_product([all_sources, all_dates], names=["fuente", "date"])

    agg = agg.set_index(["fuente", "date"]).reindex(full_index).reset_index()

    # para fuentes sin docs ese día:
    # - n_docs => 0
    # - signal_mean => 0
    # - scores => NaN (mejor que inventar 0)
    agg["n_docs"] = agg["n_docs"].fillna(0).astype(int)
    agg["signal_mean"] = agg["signal_mean"].fillna(0.0)

    # señal ponderada opcional (útil)
    #agg["sent_weighted"] = agg["sent_mean"] * agg["signal_mean"]
    agg["sent_weighted"] = agg["sent_mean"].fillna(0.0) * agg["signal_mean"]


    return agg


def agregar_macro_diario(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Combina todas las fuentes en una serie macro diaria.
    Requiere columnas: date, sent_mean, signal_mean, n_docs
    """
    d = df_daily.copy()

    # asegúrate de tipos
    d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
    d["n_docs"] = pd.to_numeric(d["n_docs"], errors="coerce").fillna(0)

    # peso por señal y volumen
    d["w"] = d["n_docs"] * d["signal_mean"].fillna(0.0)

    def wavg(g):
        wsum = g["w"].sum()
        if wsum <= 0:
            return np.nan
        return (g["sent_mean"].fillna(0.0) * g["w"]).sum() / wsum

    out = (
    d.groupby("date", as_index=False)
     .apply(
         lambda g: pd.Series({
             "macro_sent": wavg(g),
             "macro_signal": (g["signal_mean"].fillna(0.0) * g["n_docs"]).sum() / max(g["n_docs"].sum(), 1),
             "macro_docs": int(g["n_docs"].sum()),
             "macro_wsum": float(g["w"].sum()),
         }),
         include_groups=False
     )
     .reset_index(drop=True)
)


    return out

def agregar_macro_semanal(df_macro_daily: pd.DataFrame) -> pd.DataFrame:
    d = df_macro_daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])

    d["week"] = d["date"].dt.to_period("W-FRI")

    def wavg(g):
        w = g["macro_wsum"]
        if w.sum() == 0:
            return np.nan
        return (g["macro_sent"] * w).sum() / w.sum()

    out = (
        d.groupby("week")
         .apply(lambda g: pd.Series({
             "macro_sent_w": wavg(g),
             "macro_signal_w": g["macro_signal"].mean(),
             "macro_docs_w": int(g["macro_docs"].sum()),
             "macro_days": int((g["macro_wsum"] > 0).sum()),
         }),

         include_groups=False
         )
         .reset_index()
    )

    # pasamos week → fecha (viernes)
    out["date"] = out["week"].dt.to_timestamp()
    out = out.drop(columns=["week"])

    return out

import pandas as pd
import numpy as np

def to_weekly_wfri_market(df_mkt: pd.DataFrame,
                          date_col: str = "date",
                          price_col: str = "close") -> pd.DataFrame:
    """
    Convierte una serie diaria (date, close) a semanal W-FRI usando último dato disponible de la semana.
    """
    d = df_mkt.copy()
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    d = d.dropna(subset=[date_col, price_col]).sort_values(date_col)

    # semana financiera (cierre viernes). Usamos 'to_period' y luego 'last' por semana.
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    d[date_col] = d[date_col].dt.tz_convert(None)   # 👈 quitar tz explícitamente

    d["week"] = d[date_col].dt.to_period("W-FRI")
    w = (d.groupby("week", as_index=False)
           .agg(date=(date_col, "max"), close=(price_col, "last")))

    # fecha de etiqueta: viernes (timestamp del periodo)
    w["date"] = w["week"].dt.to_timestamp()  # viernes
    w = w.drop(columns=["week"]).sort_values("date")
    return w

def add_weekly_returns(df_weekly: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Añade retornos semanales: ret_1w (log o simple) y target futuro opcional.
    """
    d = df_weekly.copy().sort_values("date")
    d["ret_1w"] = d[price_col].pct_change()
    # target a 1 semana vista (lo que quieres predecir)
    d["ret_fwd_1w"] = d["ret_1w"].shift(-1)
    return d

def merge_macro_with_market(df_macro_w: pd.DataFrame, df_mkt_w: pd.DataFrame) -> pd.DataFrame:
    """
    Une macro semanal con mercado semanal por date (viernes).
    """
    m = df_macro_w.copy()
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    mk = df_mkt_w.copy()
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce")

    out = pd.merge(mk, m, on="date", how="left")

    # Si una semana no hay macro (NaN), puedes:
    # - dejar NaN (honesto)
    # - o forward-fill limitado (opcional, para modelos)
    return out

def merge_macro_with_market_daily(df_macro_d: pd.DataFrame, df_ibex_d: pd.DataFrame) -> pd.DataFrame:
    """
    Une macro diario con mercado diario por date.
    Espera:
      - df_macro_d: columnas ['date','macro_sent','macro_signal','macro_docs','macro_wsum']
      - df_ibex_d: columnas ['date','ibex_ret_d','n_stocks'] (u otras)
    Devuelve: df con mercado + macro alineado por día.
    """
    m = df_macro_d.copy()
    mk = df_ibex_d.copy()

    # Normaliza 'date' a fecha (sin hora, sin tz)
    m["date"] = pd.to_datetime(m["date"], errors="coerce").dt.date
    mk["date"] = pd.to_datetime(mk["date"], errors="coerce").dt.date

    m = m.dropna(subset=["date"])
    mk = mk.dropna(subset=["date"])

    # Convierte a datetime64[ns] (naive) para merge limpio
    m["date"] = pd.to_datetime(m["date"])
    mk["date"] = pd.to_datetime(mk["date"])

    out = pd.merge(mk, m, on="date", how="left")

    # Opcional: marcar días sin macro (honesto)
    out["has_macro"] = out["macro_docs"].fillna(0).astype(int) > 0

    return out.sort_values("date").reset_index(drop=True)


