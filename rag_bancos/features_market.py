import logging
from pathlib import Path
import pandas as pd

from rag_bancos.config import (
    BASE_FEATURES_DIR,
    SENTIMENT_FEATURES_DIR,
    BASE_DIR,
    DB_PATH,
)

logger = logging.getLogger(__name__)

from rag_bancos.market_data import load_stock_market_prices, build_ibex_proxy_daily
from rag_bancos.sentiment_aggregation import agregar_macro_semanal, merge_macro_with_market, merge_macro_with_market_daily

MARKET_FEATURES_DIR = BASE_FEATURES_DIR / "market"



# ejecutar python -m rag_bancos.features_market

def quality_filter_market_weeks(df: pd.DataFrame, min_days: int = 4, min_stocks: int = 12) -> pd.DataFrame:
    """
    Filtra semanas con proxy de mercado poco fiable:
      - min_days: mínimo de días con retorno dentro de la semana (calidad temporal)
      - min_stocks: mínimo de número típico de acciones contribuyendo (representatividad)
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    # aplica filtros solo si existen las columnas
    if "ibex_days" in out.columns:
        out = out[out["ibex_days"] >= min_days]
    if "ibex_nstocks" in out.columns:
        out = out[out["ibex_nstocks"] >= min_stocks]

    return out.sort_values("date").reset_index(drop=True)

def quality_filter_market_days(
    df: pd.DataFrame,
    min_stocks: int = 12
) -> pd.DataFrame:
    """
    Filtra días con proxy de mercado poco representativo.
    - min_stocks: mínimo de acciones contribuyendo al proxy IBEX
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])

    if "n_stocks" in out.columns:
        out = out[out["n_stocks"] >= min_stocks]

    return out.sort_values("date").reset_index(drop=True)

def quality_filter_macro_days(
    df: pd.DataFrame,
    min_docs: int = 2,
    min_signal: float = 0.1
) -> pd.DataFrame:
    """
    Filtra días con sentimiento macro poco informativo.
    """
    out = df.copy()

    if "macro_docs" in out.columns:
        out = out[out["macro_docs"] >= min_docs]

    if "macro_signal" in out.columns:
        out = out[out["macro_signal"] >= min_signal]

    return out.sort_values("date").reset_index(drop=True)

def corr_by_lags(
    df: pd.DataFrame,
    y_col: str = "ibex_ret_1w",
    x_col: str = "macro_sent_w",
    max_lag: int = 4
) -> pd.DataFrame:
    """
    Calcula correlación y_col vs x_col con lags 0..max_lag.
    Devuelve tabla con Pearson y Spearman + nº observaciones usadas.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)

    # crea lags
    for k in range(max_lag + 1):
        d[f"{x_col}_lag{k}"] = d[x_col].shift(k)

    rows = []
    for k in range(max_lag + 1):
        a = d[[y_col, f"{x_col}_lag{k}"]].dropna()
        if len(a) < 3:
            pearson = float("nan")
            spearman = float("nan")
        else:
            pearson = a.corr(method="pearson").iloc[0, 1]
            spearman = a.corr(method="spearman").iloc[0, 1]

        rows.append({
            "lag_weeks": k,
            "pearson": pearson,
            "spearman": spearman,
            "n_obs": int(len(a)),
        })

    return pd.DataFrame(rows)

def corr_by_lags_precomputed(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    max_lag: int = 5
) -> pd.DataFrame:
    d = df.copy().sort_values("date").reset_index(drop=True)

    rows = []
    # lag 0 usa x_col tal cual
    for k in range(max_lag + 1):
        xk = x_col if k == 0 else f"{x_col}_lag{k}"
        if xk not in d.columns:
            rows.append({"lag_days": k, "pearson": float("nan"), "spearman": float("nan"), "n_obs": 0})
            continue

        a = d[[y_col, xk]].dropna()
        if len(a) < 3:
            pearson = float("nan"); spearman = float("nan")
        else:
            pearson = a.corr(method="pearson").iloc[0, 1]
            spearman = a.corr(method="spearman").iloc[0, 1]

        rows.append({"lag_days": k, "pearson": pearson, "spearman": spearman, "n_obs": int(len(a))})

    return pd.DataFrame(rows)


def add_rolling_corr(
    df: pd.DataFrame,
    y_col: str = "ibex_ret_1w",
    x_col: str = "macro_sent_w",
    window: int = 26
) -> pd.DataFrame:
    """
    Añade correlación rolling (Pearson) sobre ventana de 'window' semanas.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    d[f"rolling_corr_{window}w"] = d[y_col].rolling(window).corr(d[x_col])
    return d

def add_daily_lags(
    df: pd.DataFrame,
    cols=("macro_sent", "macro_signal"),
    max_lag: int = 3
) -> pd.DataFrame:
    out = df.copy().sort_values("date").reset_index(drop=True)

    for c in cols:
        if c not in out.columns:
            continue
        for k in range(1, max_lag + 1):
            out[f"{c}_lag{k}"] = out[c].shift(k)

    return out

def add_rolling_corr_daily(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    window: int = 21,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """
    Añade correlación rolling (Pearson) diaria entre y_col y x_col.
    """
    d = df.copy().sort_values("date").reset_index(drop=True)
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    if min_periods is None:
        min_periods = window

    colname = f"rolling_corr_{window}d_{x_col}"
    d[colname] = (
        d[y_col]
        .rolling(window, min_periods=min_periods)
        .corr(d[x_col])
    )
    return d

def main():
    # 1) Cargar macro DAILY
    macro_daily_path = SENTIMENT_FEATURES_DIR / "sentiment_macro_daily.csv"
    if not macro_daily_path.exists():
        raise FileNotFoundError(f"No existe: {macro_daily_path}. Ejecuta primero sentimiento_bancos.py")

    df_macro_d = pd.read_csv(macro_daily_path)
    logger.info("Macro daily leida: %d filas", len(df_macro_d))

    # 2) Construir IBEX proxy DAILY desde SQLite (ya lo tienes en market_data)
    # 2) Mercado: extraer precios diarios desde SQLite
    df_px = load_stock_market_prices(DB_PATH)

    # 3) Construir proxy IBEX diario (media de retornos cross-section)
    df_ibex_d = build_ibex_proxy_daily(df_px)
    logger.info("IBEX proxy daily (ultimas 5):\n%s", df_ibex_d.tail(5).to_string(index=False))

    # 3) Merge DAILY
    df_daily_merged = merge_macro_with_market_daily(df_macro_d, df_ibex_d)

    out_path = MARKET_FEATURES_DIR / "macro_market_daily.csv"
    df_daily_merged.to_csv(out_path, index=False, encoding="utf-8")
    logger.info("Guardado %s", out_path.name)

    # 4) clean DAILY
    df_q = df_daily_merged.copy()

    df_q = quality_filter_market_days(df_q, min_stocks=12)
    #df_q = quality_filter_macro_days(df_q, min_docs=2, min_signal=0.1)

    df_q = add_daily_lags(df_q, max_lag=5)

    df_q = add_rolling_corr_daily(
        df_q,
        y_col="ibex_ret_d",
        x_col="macro_sent_lag2",
        window=21,
        min_periods=15
    )

    df_q = add_rolling_corr_daily(
        df_q,
        y_col="ibex_ret_d",
        x_col="macro_sent_lag3",
        window=21,
        min_periods=15
    )

    # 5) Sanity check rápido
    cols = ["ibex_ret_d", "macro_sent", "macro_signal"]
    if all(c in df_q.columns for c in cols):
        corr = df_q[cols].corr()
        logger.info("Correlacion DAILY (simple):\n%s", corr.to_string())
    else:
        logger.warning("No encuentro todas las columnas para correlacion: %s", cols)

    logger.info("Merge DAILY (ultimas 10):\n%s", df_q.tail(10).to_string(index=False))

    # 6) Correlación diaria por lags (macro_sent -> ibex_ret_d)
    logger.info("Correlacion DAILY por lags (macro_sent -> ibex_ret_d):")
    lags_sent = corr_by_lags_precomputed(
        df=df_q,
        y_col="ibex_ret_d",
        x_col="macro_sent",
        max_lag=5
    )
    logger.info("%s", lags_sent.to_string(index=False))

    # 8) Correlación diaria por lags (macro_signal -> ibex_ret_d)
    logger.info("Correlacion DAILY por lags (macro_signal -> ibex_ret_d):")
    lags_signal = corr_by_lags_precomputed(
        df=df_q,
        y_col="ibex_ret_d",
        x_col="macro_signal",
        max_lag=5
    )
    logger.info("%s", lags_signal.to_string(index=False))

    cols_show = ["date", "ibex_ret_d", "macro_sent_lag2", "rolling_corr_21d_macro_sent_lag2"]
    logger.info("Sample:\n%s", df_q[cols_show].tail(30).to_string(index=False))

if __name__ == "__main__":
    main()

