# rag_bancos/pipelines/run_models.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
)

# Importamos tu loader actual (ya funciona)
from rag_bancos.models_regression import load_dataset


Mode = Literal["daily_reg", "event_reg", "event_cls"]
Market = Literal["proxy", "index"]


@dataclass
class RunInfo:
    mode: str
    market: str
    horizon: int
    n_rows: int
    out_dir: str
    plot_path: str
    results_path: str


def _outputs_root() -> Path:
    # .../rag_bancos/pipelines/run_models.py -> parents[1] = .../rag_bancos (paquete)
    return Path(__file__).resolve().parent.parent / "outputs" / "latest"
'''
usa productos en la forma ret_fwd_H(t) = (1+r1)(1+r2)...(1+rH) - 1
conceptualmente es lo mismo que usar sumas
Esto es una aproximación lineal que funciona bien cuando los retornos diarios son 
pequeños (y normalmente lo son: ±1% típico).
Lo “financieramente correcto”: composición (producto)

'''
# usa productos en la forma ret_fwd_H(t) = (1+r1)(1+r2)...(1+rH) - 1
# conceptualmente es lo mismo que usar sumas
# Esto es una aproximación lineal que funciona bien cuando los retornos diarios son pequeños (y normalmente lo son: ±1% típico).
def _forward_return(df: pd.DataFrame, ret_col: str, horizon: int) -> pd.Series:
    """
    Retorno acumulado forward H días:
      ret_fwd(t) = Π(1+ret_{t+1..t+H}) - 1
    """
    r = df[ret_col].astype(float)
    out = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i + horizon >= len(df):
            out.iloc[i] = np.nan
            continue
        window = r.iloc[i + 1 : i + 1 + horizon]
        if window.isna().any():
            out.iloc[i] = np.nan
        else:
            out.iloc[i] = float(np.prod(1.0 + window.values) - 1.0)
    return out


def _temporal_split(df: pd.DataFrame, train_frac: float = 0.7):
    n = len(df)
    split = int(n * train_frac)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    return train, test


def _save_json(path: Path, obj: Dict[str, Any]):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def run_model(
    mode: Mode,
    market: Market = "index",
    horizon: int = 3,
    train_frac: float = 0.7,
) -> Dict[str, Any]:
    """
    Ejecuta un caso (mode) y guarda outputs estandarizados:
      outputs/latest/<mode>/plot.png
      outputs/latest/<mode>/results.json
    Devuelve un dict con rutas y resumen.
    """
    use_index = (market == "index")
    df = load_dataset(use_index=use_index)

    # Selección de columna target según mercado
    ret_col = "ibex_ret_d_index" if market == "index" else "ibex_ret_d"

    # Salidas
    out_dir = _outputs_root() / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "plot.png"
    results_path = out_dir / "results.json"

    # Asegurar fechas ordenadas
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # ============================================================
    # MODE 1: DAILY REG (lags 2-3)
    # ============================================================
    if mode == "daily_reg":
        out = df.copy()

        # Solo días con macro (igual que venías haciendo)
        out = out[out["has_macro"] == True].copy()
        out = out.sort_values("date").reset_index(drop=True)

        # Lags sobre macro_sent (2 y 3)
        out["macro_sent_lag2"] = out["macro_sent"].shift(2)
        out["macro_sent_lag3"] = out["macro_sent"].shift(3)

        # Target
        out["y"] = out[ret_col]

        # Selección final
        out = out[["date", "y", "macro_sent_lag2", "macro_sent_lag3"]].dropna().reset_index(drop=True)

        train, test = _temporal_split(out, train_frac=train_frac)
        X_train = train[["macro_sent_lag2", "macro_sent_lag3"]]
        y_train = train["y"]
        X_test = test[["macro_sent_lag2", "macro_sent_lag3"]]
        y_test = test["y"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        directional = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

        coefs = {
            "macro_sent_lag2": float(model.coef_[0]),
            "macro_sent_lag3": float(model.coef_[1]),
        }
        intercept = float(model.intercept_)

        # Plot: efecto lag2 manteniendo lag3 constante (mediana)
        lag3_const = float(out["macro_sent_lag3"].median())
        x = out["macro_sent_lag2"].values
        y = out["y"].values

        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 120)
        y_line = intercept + coefs["macro_sent_lag2"] * x_line + coefs["macro_sent_lag3"] * lag3_const

        plt.figure()
        plt.scatter(x, y)
        plt.plot(x_line, y_line)
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.xlabel("macro_sent_lag2")
        plt.ylabel(ret_col)
        plt.title("Daily: efecto macro_sent_lag2 (lag3 constante)")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

        results = {
            "mode": mode,
            "market": market,
            "ret_col": ret_col,
            "n_rows": int(len(out)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": directional},
            "coef": coefs,
            "intercept": intercept,
        }
        _save_json(results_path, results)

    # ============================================================
    # MODE 2: EVENT REG (macro_sent -> ret_fwd_Hd)
    # ============================================================
    elif mode == "event_reg":
        out = df.copy()
        out["ret_fwd"] = _forward_return(out, ret_col=ret_col, horizon=horizon)

        # Quedarnos con eventos
        out = out[out["has_macro"] == True].copy()
        out = out.sort_values("date").reset_index(drop=True)

        out = out[["date", "macro_sent", "ret_fwd"]].dropna().reset_index(drop=True)

        train, test = _temporal_split(out, train_frac=train_frac)
        X_train = train[["macro_sent"]]
        y_train = train["ret_fwd"]
        X_test = test[["macro_sent"]]
        y_test = test["ret_fwd"]

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = float(r2_score(y_test, y_pred))
        directional = float(np.mean(np.sign(y_pred) == np.sign(y_test)))

        coef = float(model.coef_[0])
        intercept = float(model.intercept_)

        # Plot
        x = out["macro_sent"].values
        y = out["ret_fwd"].values
        x_line = np.linspace(np.nanmin(x), np.nanmax(x), 120)
        y_line = intercept + coef * x_line

        plt.figure()
        plt.scatter(x, y)
        plt.plot(x_line, y_line)
        plt.axhline(0, linestyle="--", alpha=0.5)
        plt.xlabel("macro_sent (día evento)")
        plt.ylabel(f"retorno acumulado forward {horizon}d ({ret_col})")
        plt.title(f"Event-based: macro_sent -> retorno forward {horizon}d")
        plt.tight_layout()
        print("plot_path:", plot_path,"\n")
        plt.savefig(plot_path, dpi=160)
        plt.close()

        results = {
            "mode": mode,
            "market": market,
            "ret_col": ret_col,
            "horizon": horizon,
            "n_rows": int(len(out)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "metrics": {"mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": directional},
            "coef": {"macro_sent": coef},
            "intercept": intercept,
        }
        _save_json(results_path, results)

    # ============================================================
    # MODE 3: EVENT CLS (macro_sent -> P(ret_fwd>0))
    # ============================================================
    elif mode == "event_cls":
        out = df.copy()
        out["ret_fwd"] = _forward_return(out, ret_col=ret_col, horizon=horizon)

        out = out[out["has_macro"] == True].copy()
        out = out.sort_values("date").reset_index(drop=True)
        out = out[["date", "macro_sent", "ret_fwd"]].dropna().reset_index(drop=True)

        out["y_cls"] = (out["ret_fwd"] > 0).astype(int)

        train, test = _temporal_split(out, train_frac=train_frac)
        X_train = train[["macro_sent"]]
        y_train = train["y_cls"]
        X_test = test[["macro_sent"]]
        y_test = test["y_cls"]

        clf = LogisticRegression()  # default L2 (y sin warnings)
        clf.fit(X_train, y_train)

        p_up = clf.predict_proba(X_test)[:, 1]
        y_pred = (p_up >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()

        coef = float(clf.coef_[0][0])
        intercept = float(clf.intercept_[0])

        # Plot: curva logística
        x_all = out["macro_sent"].values
        x_line = pd.DataFrame({"macro_sent": np.linspace(np.nanmin(x_all), np.nanmax(x_all), 200)})
        p_line = clf.predict_proba(x_line)[:, 1]

        plt.figure()
        plt.scatter(out["macro_sent"].values, out["y_cls"].values)
        #plt.plot(x_line[:, 0], p_line)
        plt.plot(x_line["macro_sent"].values, p_line)
        plt.axhline(0.5, linestyle="--", alpha=0.5)
        plt.xlabel("macro_sent (día evento)")
        plt.ylabel(f"P(ret_fwd_{horizon}d > 0)")
        plt.title("Clasificación event-based: sentimiento -> prob. retorno positivo")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()

        results = {
            "mode": mode,
            "market": market,
            "ret_col": ret_col,
            "horizon": horizon,
            "n_rows": int(len(out)),
            "train_n": int(len(train)),
            "test_n": int(len(test)),
            "class_balance": out["y_cls"].value_counts(normalize=True).to_dict(),
            "metrics": {"accuracy": acc},
            "confusion_matrix": cm,
            "coef_logodds": {"macro_sent": coef},
            "intercept": intercept,
        }
        _save_json(results_path, results)

    else:
        raise ValueError(f"Modo no soportado: {mode}")

    info = RunInfo(
        mode=mode,
        market=market,
        horizon=horizon,
        n_rows=int(results["n_rows"]),
        out_dir=str(out_dir),
        plot_path=str(plot_path),
        results_path=str(results_path),
    )
    return {"run": asdict(info), "results": results}
