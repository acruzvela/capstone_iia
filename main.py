from fastapi import FastAPI, Request, HTTPException, Path as PathParam
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator
from typing import List, Dict, Any
import sqlite3
from pathlib import Path
from agente.ibex_agent import run_ibex_agent
from fastapi.responses import FileResponse
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import os
import logging
import re
import rag_bancos

from rag_bancos.pipelines.run_models import run_model
from rag_bancos.models_regression import load_dataset

import transformer_market
from transformer_market.config import OUTPUT_ROOT as TF_OUT

# -------------------------------------------------------------------
# Rutas y constantes RAG BANCOS
RAG_BANCOS_DIR = Path(rag_bancos.__file__).resolve().parent
SENTIMENT_DIR = RAG_BANCOS_DIR / "data" / "features" / "sentiment"
BANCOS_DATA_DIR = RAG_BANCOS_DIR / "data" / "bancos"

ALLOWED_EXTS = {".txt", ".md", ".json", ".html", ".htm"}

# -------------------------------------------------------------------
# Rutas y constantes TRANSFORMER MARKET
TF_DATASET_DIR = TF_OUT / "dataset"
TF_REPORTS_DIR = TF_OUT / "reports"
TF_PLOTS_DIR = TF_REPORTS_DIR / "plots"
TF_PREDS_DIR = TF_OUT / "preds"
TF_MODELS_DIR = TF_OUT / "models"

# -------------------------------------------------------------------
# Configuración
# -------------------------------------------------------------------
# ejecutar: uvicorn main:app --reload

app = FastAPI(title="IBEX35 API v4")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Carpeta donde estarán los HTML (igual que en fastapi3)
templates = Jinja2Templates(directory="templates")

# Montar carpeta static para servir CSS y JS
static_path = Path(__file__).resolve().parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"Carpeta static montada: {static_path}")
else:
    logger.warning(f"Carpeta static no encontrada: {static_path}")

# Ruta a la base de datos SQLite
DB_PATH = Path(r"E:/cruz/informatica/sqlite/ibex35.db")


# -------------------------------------------------------------------
# Utilidad: conexión a la base de datos
# -------------------------------------------------------------------
def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------------------------------------------------
# Funciones de consulta
# -------------------------------------------------------------------
def consulta_stock_market(ticker: str, limit: int = 200) -> List[Dict[str, Any]]:
    """
    Consulta la tabla 'stock_market' filtrando por ticker.
    Devuelve las filas más recientes primero.
    """
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT date, ticker, open, high, low, close, volume
        FROM stock_market
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (ticker, limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def consulta_companies() -> List[Dict[str, Any]]:
    """
    Devuelve todas las compañías de la tabla 'companies'.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            ticker, name, isin, nominal_value, market,
            market_cap, shares_outstanding, currency,
            sector_icb, industry, free_float, country
        FROM companies
        ORDER BY ticker
        """
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def consulta_company(ticker: str) -> Dict[str, Any] | None:
    """
    Devuelve una sola compañía por ticker o None si no existe.
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT
            ticker, name, isin, nominal_value, market,
            market_cap, shares_outstanding, currency,
            sector_icb, industry, free_float, country
        FROM companies
        WHERE ticker = ?
        """,
        (ticker,),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

# -------------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------------

def sanitize_jsonable(obj):
    if isinstance(obj, dict):
        return {k: sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj

def _csv_summary(path: Path, head_n: int = 6) -> dict:
    df = pd.read_csv(path)

    # Normalizamos date si existe
    date_min = date_max = None
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], errors="coerce")
        date_min = None if d.isna().all() else d.min().strftime("%Y-%m-%d")
        date_max = None if d.isna().all() else d.max().strftime("%Y-%m-%d")

    head = df.head(head_n).copy()
    if "date" in head.columns:
        head["date"] = pd.to_datetime(head["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    payload = {
        "file": path.name,
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": df.columns.tolist(),
        "date_range": {"min": date_min, "max": date_max},
        "head": head.to_dict(orient="records"),
    }
    return sanitize_jsonable(payload)  # ya la tienes en main.py


def _safe_dt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

def _scan_docs_dir(dir_path: Path, max_samples: int = 5) -> Dict[str, Any]:
    if not dir_path.exists():
        return {"exists": False, "n_files": 0, "samples": [], "last_modified": None}

    files = []
    for p in dir_path.rglob("*"):
        if not p.is_file():
            continue

        name_l = p.name.lower()
        suf_l = p.suffix.lower()

        # ✅ excluir metadatos
        if name_l.endswith(".meta.json") or name_l in ("meta.json", "metadata.json"):
            continue

        # docs permitidos (incluye sin extensión si lo usas)
        if (suf_l in ALLOWED_EXTS) or (p.suffix == ""):
            files.append(p)

    if not files:
        return {"exists": True, "n_files": 0, "samples": [], "last_modified": None}

    files_sorted = sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)
    mtimes = [f.stat().st_mtime for f in files_sorted]

    samples = []
    for f in files_sorted[:max_samples]:
        samples.append({
            "file": f.name,
            "relpath": str(f.relative_to(dir_path)),
            "modified": _safe_dt(f.stat().st_mtime),
        })

    return {
        "exists": True,
        "n_files": len(files_sorted),
        "mtime_range": {"min": _safe_dt(min(mtimes)), "max": _safe_dt(max(mtimes))},
        "last_doc": {
            "file": files_sorted[0].name,
            "relpath": str(files_sorted[0].relative_to(dir_path)),
            "modified": _safe_dt(files_sorted[0].stat().st_mtime),
        },
        "samples": samples,
    }

# -------------------------------------------------------------------
# Frontend (HTML)
# -------------------------------------------------------------------
@app.get("/")
def index(request: Request):
    """
    Devuelve la página HTML principal.
    Puedes seguir usando tu index3.html o el nuevo index.html
    que hicimos para el grid responsive.
    """
    return templates.TemplateResponse("index.html", {"request": request})


# -------------------------------------------------------------------
# API: COMPANIES
# -------------------------------------------------------------------
@app.get("/api/companies")
def api_companies():
    """
    Devuelve la lista completa de compañías (para grids, filtros, etc.).
    """
    companies = consulta_companies()
    return {"companies": companies}


@app.get("/api/companies/{ticker}")
def api_company(ticker: str = PathParam(..., regex="^[A-Z0-9]{1,5}$")):
    """
    Devuelve los datos de una compañía concreta.
    """
    try:
        logger.info(f"[COMPANY] Consultando {ticker}")
        t = ticker.upper()
        company = consulta_company(t)
        if company is None:
            raise HTTPException(status_code=404, detail=f"Company {t} not found")
        return company
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[COMPANY] Error consultando {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error consultando compañía: {str(e)}")


@app.get("/api/tickers")
def api_tickers():
    """
    Devuelve solo la lista de tickers, sacados de la tabla 'companies'.
    (Así tu frontend puede rellenar el grid de selección).
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT ticker FROM companies ORDER BY ticker")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return {"tickers": rows}


# -------------------------------------------------------------------
# API: PRECIOS (stock_market)
# -------------------------------------------------------------------
@app.get("/api/stock/{ticker}")
def api_stock(ticker: str = PathParam(..., regex="^[A-Z0-9]{1,5}$"), limit: int = 200):
    """
    Devuelve las últimas 'limit' cotizaciones de un ticker
    desde la tabla 'stock_market'.
    """
    # Validar limit
    if limit < 1 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit debe estar entre 1 y 1000")
    
    try:
        logger.info(f"[STOCK] Consultando {ticker} (limit={limit})")
        t = ticker.upper()
        datos = consulta_stock_market(t, limit=limit)
        if not datos:
            raise HTTPException(status_code=404, detail=f"No hay datos para {t}")
        logger.info(f"[STOCK] {ticker}: {len(datos)} registros")
        return {"ticker": t, "prices": datos}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[STOCK] Error consultando {ticker}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error consultando stock: {str(e)}")


# -------------------------------------------------------------------
# API: MULTITICKERS
# -------------------------------------------------------------------

class TickersRequest(BaseModel):
    tickers: List[str]
    limit: int | None = 200  # opcional, nº de días a devolver
    
    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        """Validar que tickers sean válidos (1-20 tickers, formato A-Z0-9)."""
        if not v or len(v) > 20:
            raise ValueError("Debe haber entre 1 y 20 tickers")
        for t in v:
            if not re.match(r'^[A-Z0-9]{1,5}$', t.upper()):
                raise ValueError(f"Ticker inválido: {t}. Debe ser alfanumérico (1-5 caracteres)")
        return [x.upper() for x in v]
    
    @field_validator('limit')
    @classmethod
    def validate_limit(cls, v: int | None) -> int:
        """Validar que limit esté entre 1 y 1000."""
        if v is None:
            return 200
        if v < 1 or v > 1000:
            raise ValueError("limit debe estar entre 1 y 1000")
        return v


@app.post("/api/multi")
def api_multi(req: TickersRequest):
    """
    Recibe un JSON:
      { "tickers": ["SAN","BBVA"], "limit": 100 }

    Devuelve:
      {
        "SAN": { "company": {...}, "prices": [...] },
        "BBVA": { "company": {...}, "prices": [...] }
      }
    """
    conn = get_conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    resultados: Dict[str, Dict[str, Any]] = {}
    limit = req.limit or 200

    for t in req.tickers:
        ticker = t.upper()

        # Datos de compañía
        cur.execute(
            """
            SELECT
                ticker, name, isin, nominal_value, market,
                market_cap, shares_outstanding, currency,
                sector_icb, industry, free_float, country
            FROM companies
            WHERE ticker = ?
            """,
            (ticker,),
        )
        company_row = cur.fetchone()
        company_data = dict(company_row) if company_row else None

        # Datos de precios
        cur.execute(
            """
            SELECT date, ticker, open, high, low, close, volume
            FROM stock_market
            WHERE ticker = ?
            ORDER BY date DESC
            LIMIT ?
            """,
            (ticker, limit),
        )
        prices = [dict(r) for r in cur.fetchall()]

        resultados[ticker] = {
            "company": company_data,
            "prices": prices,
        }

    conn.close()
    return resultados

@app.get("/api/debug-count/{ticker}")
def debug_count(ticker: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM stock_market WHERE ticker = ?", (ticker,))
    n = cur.fetchone()[0]
    conn.close()
    return {"ticker": ticker, "count": n}

class UpdateRequest(BaseModel):
    tickers: List[str]

@app.post("/api/update")
def api_update(req: UpdateRequest):
    if not req.tickers:
        raise HTTPException(status_code=400, detail="No se han enviado tickers")
    
    logger.info(f"[AGENTE] Iniciando con tickers: {req.tickers}")
    try:
        texto = run_ibex_agent(req.tickers)
        logger.info(f"[AGENTE] Completado exitosamente")
        return {"message": texto}
    except Exception as e:
        logger.error(f"[AGENTE] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error ejecutando agente: {str(e)}")
    texto = run_ibex_agent(req.tickers)
    return {"message": texto}

# -------------------------------------------------------------------
# API: RAG MODE
# -------------------------------------------------------------------

@app.post("/api/run/model")
def api_run_model(mode: str, market: str = "index", horizon: int = 3):
    """
    Ejecuta un modo del capstone y genera outputs:
      - outputs/latest/<mode>/plot.png
      - outputs/latest/<mode>/results.json
    """
    return run_model(mode=mode, market=market, horizon=horizon)

@app.get("/api/pipeline/rag")
def api_pipeline_rag(max_samples: int = 5):
    """
    Resumen del 'RAG corpus': cuántos docs hay por carpeta y últimas fechas.
    """
    if not BANCOS_DATA_DIR.exists():
        raise HTTPException(status_code=500, detail=f"No existe: {BANCOS_DATA_DIR}")

    buckets = ["bce", "bde", "fed", "general_eng", "general_es"]
    out: Dict[str, Any] = {
        "kind": "pipeline_rag",
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(BANCOS_DATA_DIR),
        "buckets": {},
        "totals": {"n_files": 0},
    }

    total = 0
    for b in buckets:
        info = _scan_docs_dir(BANCOS_DATA_DIR / b, max_samples=max_samples)
        out["buckets"][b] = info
        total += int(info.get("n_files", 0) or 0)

    out["totals"]["n_files"] = total
    return out

@app.get("/api/pipeline/sentiment")
def api_pipeline_sentiment():
    """
    Devuelve un resumen del sentimiento macro agregado
    a partir de macro_market_daily.csv
    """
    try:
        logger.info("[SENTIMENT] Cargando dataset de sentimiento")
        df = load_dataset(use_index=True)

        # Solo filas con información de sentimiento
        df_sent = df[df["macro_sent"].notna()].copy()

        date_min = df_sent["date"].min()
        date_max = df_sent["date"].max()

        # Conteos útiles
        total_events = int(df["has_macro"].fillna(False).sum())
        sent_events = int(df_sent.shape[0])

        sig = pd.to_numeric(df_sent["macro_signal"], errors="coerce").dropna()

        signal_bins = {
            "fuerte (>=1.0)": int((sig >= 1.0).sum()),
            "media (0.5-0.99)": int(((sig >= 0.5) & (sig < 1.0)).sum()),
            "débil (<0.5)": int((sig < 0.5).sum()),
        }

        # Muestra pequeña
        sample_cols = [
            "date",
            "macro_sent",
            "macro_signal",
            "macro_docs",
            "macro_wsum",
        ]

        head = df_sent[sample_cols].head(8).copy()
        head["date"] = pd.to_datetime(head["date"]).dt.strftime("%Y-%m-%d")

        payload = {
            "kind": "pipeline_sentiment",
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "macro_market_daily.csv",
            "date_range": {
                "min": pd.to_datetime(date_min).strftime("%Y-%m-%d"),
                "max": pd.to_datetime(date_max).strftime("%Y-%m-%d"),
            },
            "counts": {
                "has_macro_true": total_events,
                "macro_sent_notna": sent_events,
            },
            "signal_bins": signal_bins,
            "head": head.to_dict(orient="records"),
        }
        logger.info(f"[SENTIMENT] {sent_events} eventos procesados")
        return sanitize_jsonable(payload)
    except Exception as e:
        logger.error(f"[SENTIMENT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cargando sentimiento: {str(e)}")

    return sanitize_jsonable(payload)


@app.get("/api/pipeline/features")
def api_pipeline_features(market: str = "index"):
    """
    Devuelve un resumen del dataset final (features mercado + macro),
    listo para mostrar en el frontend.
    market: "index" o "proxy"
    """
    try:
        logger.info(f"[FEATURES] Cargando dataset (market={market})")
        use_index = (market.lower() == "index")

        df = load_dataset(use_index=use_index)

        # columnas clave si existen
        date_min = df["date"].min()
        date_max = df["date"].max()

        has_macro_n = int(df["has_macro"].fillna(False).sum()) if "has_macro" in df.columns else None
        macro_sent_n = int(df["macro_sent"].notna().sum()) if "macro_sent" in df.columns else None

        # muestra pequeñita (para no devolver un monstruo)
        sample_cols = [c for c in [
            "date",
            "macro_sent",
            "macro_signal",
            "macro_docs",
            "macro_wsum",
            "has_macro",
            "ibex_ret_d",
            "ibex_ret_d_index",
        ] if c in df.columns]

        head = df[sample_cols].head(8).copy() if sample_cols else df.head(8).copy()
        head["date"] = pd.to_datetime(head["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        head = head.replace([np.inf, -np.inf], np.nan)
        head = head.where(pd.notna(head), None)


        payload = {
            "kind": "pipeline_features",
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "market": "index" if use_index else "proxy",
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "date_range": {
                "min": None if pd.isna(date_min) else pd.to_datetime(date_min).strftime("%Y-%m-%d"),
                "max": None if pd.isna(date_max) else pd.to_datetime(date_max).strftime("%Y-%m-%d"),
            },
            "counts": {
                "has_macro_true": has_macro_n,
                "macro_sent_notna": macro_sent_n,
            },
            "columns": df.columns.tolist(),
            "head": head.to_dict(orient="records"),
        }

        logger.info(f"[FEATURES] Dataset cargado: {df.shape[0]} filas")
        return sanitize_jsonable(payload)
    except Exception as e:
        logger.error(f"[FEATURES] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cargando features: {str(e)}")

@app.get("/api/pipeline/sentiment-files")
def api_pipeline_sentiment_files():
    """
    Resume los ficheros intermedios del pipeline de sentimiento:
    sentiment_docs.csv, sentiment_macro_daily.csv, ...
    """
    if not SENTIMENT_DIR.exists():
        raise HTTPException(status_code=500, detail=f"No existe: {SENTIMENT_DIR}")

    files = sorted(SENTIMENT_DIR.glob("*.csv"))
    if not files:
        return {
            "kind": "pipeline_sentiment_files",
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dir": str(SENTIMENT_DIR),
            "files": [],
        }

    summaries = [_csv_summary(p) for p in files]

    return sanitize_jsonable({
        "kind": "pipeline_sentiment_files",
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dir": str(SENTIMENT_DIR),
        "files": summaries,
    })


@app.get("/api/results/{mode}")
def api_results(mode: str):
    """
    Devuelve el JSON de resultados del último run de ese modo.
    Modes permitidos: daily_reg, event_cls, event_reg
    """
    allowed_modes = {"daily_reg", "event_cls", "event_reg"}
    if mode not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"Mode inválido: {mode}. Debe ser uno de {allowed_modes}")
    
    try:
        results_path = (
            Path(__file__).resolve().parent
            / "rag_bancos" 
            / "outputs" / "latest" / mode / "results.json"
        )
        
        # Validar path traversal
        if not results_path.resolve().is_relative_to((Path(__file__).resolve().parent / "rag_bancos" / "outputs").resolve()):
            raise HTTPException(status_code=403, detail="Acceso denegado")
        
        if not results_path.exists():
            raise HTTPException(status_code=404, detail="Resultados no encontrados. Ejecuta primero /api/run/model.")
        
        logger.info(f"[RESULTS] Leyendo {mode}")
        return json.loads(results_path.read_text(encoding="utf-8"))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[RESULTS] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error cargando resultados: {str(e)}")


@app.get("/api/plot/{mode}")
def api_plot(mode: str):
    """
    Devuelve la imagen PNG del último run de ese modo.
    """
    allowed_modes = {"daily_reg", "event_cls", "event_reg"}
    if mode not in allowed_modes:
        raise HTTPException(status_code=400, detail=f"Mode inválido: {mode}")
    
    try:
        plot_path = (
            Path(__file__).resolve().parent
            / "rag_bancos"
            / "outputs" / "latest" / mode / "plot.png"
        )
        
        # Validar path traversal
        if not plot_path.resolve().is_relative_to((Path(__file__).resolve().parent / "rag_bancos" / "outputs").resolve()):
            raise HTTPException(status_code=403, detail="Acceso denegado")
        
        if not plot_path.exists():
            raise HTTPException(status_code=404, detail="Plot no encontrado. Ejecuta primero /api/run/model.")
        
        logger.info(f"[PLOT] Sirviendo {mode}")
        return FileResponse(str(plot_path), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PLOT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error sirviendo plot: {str(e)}")
    return FileResponse(str(plot_path), media_type="image/png")

# -------------------------------------------------------------------
# API: TRANSFORMER MARKET
# -------------------------------------------------------------------

@app.get("/api/transformer/status")
def api_transformer_status():
    metrics_path = TF_REPORTS_DIR / "metrics.json"
    preds_path = TF_PREDS_DIR / "preds_test.csv"
    model_path = TF_MODELS_DIR / "transformer_regressor.pt"

    ready = metrics_path.exists() and preds_path.exists() and model_path.exists()

    return {
        "ready": ready,
        "paths": {
            "out": str(TF_OUT),
            "metrics": str(metrics_path),
            "preds": str(preds_path),
            "model": str(model_path),
            "plots_dir": str(TF_PLOTS_DIR),
        },
        "exists": {
            "metrics": metrics_path.exists(),
            "preds": preds_path.exists(),
            "model": model_path.exists(),
            "plots_dir": TF_PLOTS_DIR.exists(),
        },
    }


@app.get("/api/transformer/metrics")
def api_transformer_metrics():
    metrics_path = TF_REPORTS_DIR / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json no encontrado. Ejecuta el pipeline Transformer.")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


@app.get("/api/transformer/plot/{name}")
def api_transformer_plot(name: str):
    """
    Devuelve un plot PNG del transformer.
    Nombres permitidos: loss_train_vs_val.png, scatter_pred_vs_true.png, hist_pred_true.png
    """
    allowed = {
        "loss_train_vs_val.png",
        "scatter_pred_vs_true.png",
        "hist_pred_true.png",
    }
    if name not in allowed:
        raise HTTPException(status_code=400, detail=f"Plot no permitido: {name}")

    try:
        plot_path = TF_PLOTS_DIR / name
        
        # Validar path traversal
        if not plot_path.resolve().is_relative_to(TF_PLOTS_DIR.resolve()):
            raise HTTPException(status_code=403, detail="Acceso denegado")
        
        if not plot_path.exists():
            raise HTTPException(status_code=404, detail="Plot no encontrado. Ejecuta evaluate.py.")
        
        logger.info(f"[TF-PLOT] Sirviendo {name}")
        return FileResponse(str(plot_path), media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[TF-PLOT] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error sirviendo plot: {str(e)}")

@app.get("/api/transformer/preds")
def api_transformer_preds(limit: int = 50):
    """
    Devuelve últimas predicciones del test (por fecha).
    Columns esperadas: date, y_true, y_pred
    """
    path = TF_PREDS_DIR / "preds_test.csv"
    if not path.exists():
        raise HTTPException(status_code=404, detail="preds_test.csv no encontrado.")

    df = pd.read_csv(path)

    # normaliza date y ordena
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")

        # formatea date para JSON
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # columnas y métricas auxiliares
    for c in ["y_true", "y_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if ("y_true" in df.columns) and ("y_pred" in df.columns):
        df["error"] = df["y_pred"] - df["y_true"]
        df["abs_error"] = (df["error"]).abs()

    # nos quedamos con las últimas N
    df_out = df.tail(int(limit)).copy()

    # sanea NaN/inf
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    rows = df_out.where(pd.notna(df_out), None).to_dict(orient="records")

    return {
        "file": str(path),
        "n_rows_total": int(len(df)),
        "limit": int(limit),
        "rows": rows,
        "columns": df_out.columns.tolist(),
    }
