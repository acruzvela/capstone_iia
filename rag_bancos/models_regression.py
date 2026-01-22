import logging
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa
import sqlite3

from rag_bancos.config import DB_PATH
from rag_bancos.features_market import MARKET_FEATURES_DIR
from rag_bancos.features_market import quality_filter_macro_days, quality_filter_market_days

# ejecutar python -m rag_bancos.models_regression

logger = logging.getLogger(__name__)

def _ensure_columns(df: pd.DataFrame, cols: list) -> None:
    """Valida que existan columnas obligatorias."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en DataFrame: {missing}")

def load_ibex_index(db_path: str, table: str = "ibex_index") -> pd.DataFrame:
    """
    Lee ibex_index desde SQLite y devuelve DF con date y close (y opcionalmente OHLCV).
    """
    sql = f"""
    SELECT date, open, high, low, close, volume
    FROM {table}
    ORDER BY date
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def load_dataset(
    csv_path: str = MARKET_FEATURES_DIR / "macro_market_daily.csv",
    db_path: str = DB_PATH,
    use_index: bool = True,
    ) -> pd.DataFrame:
    """Carga dataset macro + mercado diario con validaciones."""
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    
    if df.empty:
        logger.warning("load_dataset: CSV vacio")

    if use_index:
        df_index = load_ibex_index(db_path=db_path, table="ibex_index")
        df_index = add_index_returns(df_index)

        # nos quedamos con lo mínimo para mergear
        df_index = df_index[["date", "close", "ibex_ret_d_index"]].rename(
            columns={"close": "ibex_close_index"}
        )

        # merge 1-a-1 por fecha
        df = df.merge(df_index, on="date", how="left")

    return df

def add_index_returns(df_index: pd.DataFrame) -> pd.DataFrame:
    """
    Añade retorno diario del índice basado en close:
      ibex_ret_d_index = close.pct_change()
    """
    out = df_index.copy()
    out["ibex_ret_d_index"] = out["close"].pct_change()
    # opcional: log-return
    # out["ibex_logret_d_index"] = np.log(out["close"]).diff()

    return out




def prepare_dataset(df: pd.DataFrame, target_col: str = "ibex_ret_d") -> pd.DataFrame:
    """
    Limpia y prepara dataset para modelado:
    - filtra calidad mercado/macro
    - crea lags en variables macro (1,2,3)
    - deja un df final sin NaNs en X/y
    
    Args:
        df: DF con columnas: date, macro_sent, macro_signal, ibex_ret_d, has_macro, n_stocks
        target_col: nombre de columna target (default "ibex_ret_d")
    
    Returns:
        pd.DataFrame limpio con columns: date, y_ret_d, macro_sent_lag2, macro_sent_lag3, macro_signal_lag2, macro_signal_lag3
    """

    out = df.copy()

    # 1) asegurar fechas y orden
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # 2) filtros de calidad (mercado)
    out = quality_filter_market_days(out, min_stocks=12)

    # 3) quedarnos SOLO con días con macro disponible
    # (si quieres permitir forward-fill luego, lo haremos más adelante, pero para regresión simple: no)
    out = out[out["has_macro"] == True].copy()

    # 4) filtros de calidad macro (ajústalos si te dejan muy pocos datos)
    # si en tu dataset hay pocos días con macro, pon min_docs=1 para empezar
    #out = quality_filter_macro_days(out, min_docs=1, min_signal=0.1)

    # 5) crear lags que ya vimos interesantes
    # (si ya existen en el CSV, esto los reescribe; no pasa nada)
    out = out.sort_values("date").reset_index(drop=True)
    for k in (1, 2, 3):
        out[f"macro_sent_lag{k}"] = out["macro_sent"].shift(k)
        out[f"macro_signal_lag{k}"] = out["macro_signal"].shift(k)

    # 6) target (regresión): retorno diario
    out["y_ret_d"] = out[target_col]
    #out["y_ret_d"] = out["ibex_ret_d"]

    # 7) selección final de columnas para modelado
    feature_cols = [
        "macro_sent_lag2",
        "macro_sent_lag3",
        # opcional: añade señal como feature
        "macro_signal_lag2",
        "macro_signal_lag3",
    ]

    keep = ["date", "y_ret_d"] + feature_cols
    out = out[keep].copy()

    # 8) drop NaNs (lags crean NaN al principio, y también habrá días sin macro_sent)
    out = out.dropna().reset_index(drop=True)

    logger.info("Dataset modelado listo: %d filas", len(out))
    logger.info("Rango modelado: %s -> %s", out["date"].min().date() if len(out) > 0 else "N/A", out["date"].max().date() if len(out) > 0 else "N/A")
    logger.info("Features: %s", feature_cols)

    return out

def temporal_split(df, train_frac=0.7):
    n = len(df)
    split = int(n * train_frac)

    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()

    X_train = train[["macro_sent_lag2", "macro_sent_lag3"]]
    y_train = train["y_ret_d"]

    X_test = test[["macro_sent_lag2", "macro_sent_lag3"]]
    y_test = test["y_ret_d"]

    return X_train, X_test, y_train, y_test

def temporal_split_delta(df, train_frac=0.7):
    n = len(df)
    split = int(n * train_frac)

    train = df.iloc[:split].copy()
    test  = df.iloc[split:].copy()

    X_train = train[["delta_sent"]]
    y_train = train["y_ret_d"]

    X_test = test[["delta_sent"]]
    y_test = test["y_ret_d"]

    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)

    # directional accuracy (evita líos con ceros)
    sign_true = np.sign(y_test.to_numpy())
    sign_pred = np.sign(np.asarray(y_pred))
    sign_acc = float(np.mean(sign_true == sign_pred))

    results = {
        "MAE": float(mae),
        "RMSE": rmse,
        "R2": float(r2),
        "Directional_Accuracy": sign_acc
    }

    return y_pred, results

def main():

    df = load_dataset()
    df = prepare_dataset(df, target_col="ibex_ret_d_index")
    X_train, X_test, y_train, y_test = temporal_split(df)
    model = train_linear_regression(X_train, y_train)
    y_pred, results=evaluate_regression(model, X_test, y_test)
    print("📊 Métricas:")
    for k, v in results.items():
        print(f"{k:>20}: {v:.4f}")

    print("\n📐 Coeficientes:")
    coef_df = pd.Series(
        model.coef_,
        index=X_train.columns,
        name="coef"
    )
    print(coef_df)

    print(f"\nIntercept: {model.intercept_:.6f}")

    print("\n📈 Predicciones vs reales:")
    eval_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred
    }, index=y_test.index)

    print(eval_df)

    # rango de sentimiento
    x = np.linspace(-1, 1, 100)

    # fijamos lag3 en su media
    lag3_mean = X_train["macro_sent_lag3"].mean()

    # predicción "teórica"
    y_line = (
        model.intercept_
        + model.coef_[0] * x
        + model.coef_[1] * lag3_mean
    )

    plt.figure()
    plt.scatter(X_train["macro_sent_lag2"], y_train)
    plt.plot(x, y_line)
    plt.xlabel("macro_sent_lag2")
    plt.ylabel("ibex_ret_d_index")
    plt.title("Efecto de macro_sent_lag2 manteniendo lag3 constante")
    plt.show()

    # datos
    X = df[["macro_sent_lag2", "macro_sent_lag3"]]
    y = df["y_ret_d"]

    x_vals = X["macro_sent_lag2"].values
    y_vals = X["macro_sent_lag3"].values
    z_vals = y.values

    # malla para el plano
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_vals.min(), x_vals.max(), 20),
        np.linspace(y_vals.min(), y_vals.max(), 20)
    )

    z_plane = (
        model.intercept_
        + model.coef_[0] * x_grid
        + model.coef_[1] * y_grid
    )

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(x_vals, y_vals, z_vals)
    ax.plot_surface(x_grid, y_grid, z_plane, alpha=0.5)

    ax.set_xlabel("macro_sent_lag2")
    ax.set_ylabel("macro_sent_lag3")
    ax.set_zlabel("ibex_ret_d_index")

    ax.set_title("Plano de regresión: sentimiento macro → retorno IBEX")

    plt.show()

if __name__ == "__main__":
    main()

