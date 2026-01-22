# transformer_market/eval/evaluate.py

'''
Responsabilidad del archivo (muy clara)
evaluate.py debe:
    Leer preds_test.csv
    Calcular métricas numéricas
    Calcular métricas direccionales
    Generar gráficos simples
    Guardar resultados en:
        metrics.json
        plots/*.png

Métricas que vamos a calcular (defendibles)
    Regresión
        MAE
        RMSE
        R² (con cautela, pero se pide mucho)
    Dirección (muy importante en mercados)  
        Directional Accuracy
            1[sign(y_pred)=sign(y_true)]
    Baseline (obligatorio)
        Baseline ingenuo: y_pred = 0
            directional accuracy ≈ % de días positivos
            MAE = media(|y_true|)

'''



from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from transformer_market.config import OUTPUT_ROOT

logger = logging.getLogger(__name__)

# ejecutar python -m transformer_market.eval.evaluate


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula directional accuracy (% de veces que signo coincide).
    
    Métrica importante para trading: predecir dirección es más relevante 
    que predicción exacta de magnitud.
    
    Args:
        y_true: array de retornos reales
        y_pred: array de predicciones
    
    Returns:
        float en [0, 1]: porcentaje de aciertos en dirección
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"Shapes mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    if len(y_true) == 0:
        raise ValueError("Arrays vacíos")
    
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def evaluate_predictions(
    preds_path: Path | str,
    output_dir: Path | str,
) -> Path:
    """
    Calcula métricas de regresión y dirección, genera plots.
    
    Métricas:
        - MAE, RMSE, R²: regresión
        - Directional Accuracy: % aciertos en dirección
        - Baseline (predicción cero): para comparación
    
    Genera:
        - metrics.json: métricas numéricas
        - plots/scatter_pred_vs_true.png: scatter plot
        - plots/hist_pred_true.png: histograma
    
    Args:
        preds_path: Path o str a preds_test.csv (columnas: y_true, y_pred)
        output_dir: Path o str para guardar resultados
    
    Returns:
        Path a metrics.json generado
        
    Raises:
        FileNotFoundError: si preds_path no existe
        ValueError: si DF vacío o columnas faltantes
    """
    preds_path = Path(preds_path)
    output_dir = Path(output_dir)
    
    if not preds_path.exists():
        raise FileNotFoundError(f"Archivo de predicciones no encontrado: {preds_path}")
    
    logger.info(f"Cargando predicciones desde: {preds_path}")
    
    df = pd.read_csv(preds_path)
    
    if df.empty:
        raise ValueError("DataFrame de predicciones está vacío")
    
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError(f"Faltan columnas requeridas. Encontradas: {df.columns.tolist()}")
    
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    
    logger.info(f"Predicciones cargadas: {len(y_true)} muestras")

    # ===== Métricas modelo =====
    logger.info("Calculando métricas...")
    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(directional_accuracy(y_true, y_pred)),
    }
    
    logger.info(f"Modelo - MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}, R²: {metrics['r2']:.6f}, DA: {metrics['directional_accuracy']:.4f}")

    # ===== Baseline ingenuo: predicción cero =====
    y_base = np.zeros_like(y_true)

    baseline = {
        "mae": float(mean_absolute_error(y_true, y_base)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_base))),
        "directional_accuracy": float(directional_accuracy(y_true, y_base)),
    }
    
    logger.info(f"Baseline - MAE: {baseline['mae']:.6f}, RMSE: {baseline['rmse']:.6f}, DA: {baseline['directional_accuracy']:.4f}")

    results = {
        "model": metrics,
        "baseline_zero": baseline,
        "n_test_samples": int(len(y_true)),
    }

    # ===== Guardar métricas =====
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Métricas guardadas en: {metrics_path}")

    # ===== Gráficos =====
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generando plots...")

    # --- Scatter y_true vs y_pred ---
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lim = max(np.max(np.abs(y_true)), np.max(np.abs(y_pred)))
    plt.plot([-lim, lim], [-lim, lim], "r--", label="y=x")
    plt.xlabel("y_true (log return P5)")
    plt.ylabel("y_pred")
    plt.title("Predicciones vs Real")
    plt.legend()
    plt.tight_layout()
    scatter_path = plots_dir / "scatter_pred_vs_true.png"
    plt.savefig(scatter_path)
    plt.close()
    logger.debug(f"Scatter plot guardado: {scatter_path}")

    # --- Distribución de predicciones ---
    plt.figure(figsize=(6, 4))
    plt.hist(y_pred, bins=30, alpha=0.7, label="Pred")
    plt.hist(y_true, bins=30, alpha=0.7, label="True")
    plt.legend()
    plt.title("Distribución y_true vs y_pred")
    plt.xlabel("Log Return")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    hist_path = plots_dir / "hist_pred_true.png"
    plt.savefig(hist_path)
    plt.close()
    logger.debug(f"Histogram plot guardado: {hist_path}")

    logger.info(f"Evaluación completada. Plots guardados en: {plots_dir}")

    return metrics_path

def plot_training_history(history_path: Path | str, output_dir: Path | str) -> Path:
    """
    Genera gráfico de pérdida de entrenamiento vs validación.
    
    Args:
        history_path: Path o str a train_history.json
        output_dir: Path o str para guardar plot
    
    Returns:
        Path a loss plot generado
        
    Raises:
        FileNotFoundError: si history_path no existe
        ValueError: si JSON corrupto o faltan claves
    """
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    
    if not history_path.exists():
        raise FileNotFoundError(f"Archivo de history no encontrado: {history_path}")
    
    logger.info(f"Cargando historia de entrenamiento desde: {history_path}")
    
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON corrupto en {history_path}: {e}") from e
    
    if "train_loss" not in history or "val_loss" not in history:
        raise ValueError(f"Faltan claves esperadas en history.json: {history.keys()}")

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    
    if len(train_loss) != len(val_loss):
        raise ValueError(f"Mismatch en longitud: train_loss={len(train_loss)}, val_loss={len(val_loss)}")
    
    epochs = range(1, len(train_loss) + 1)
    
    logger.info(f"Historia cargada: {len(epochs)} epochs")

    plots_dir = output_dir / "reports" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, train_loss, label="Train loss", marker="o", markersize=3)
    plt.plot(epochs, val_loss, label="Validation loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Huber)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    loss_path = plots_dir / "loss_train_vs_val.png"
    plt.savefig(loss_path)
    plt.close()
    
    logger.info(f"Gráfico de pérdidas guardado: {loss_path}")
    
    return loss_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    
    preds_path = OUTPUT_ROOT / "preds" / "preds_test.csv"
    history_path = OUTPUT_ROOT / "reports" / "train_history.json"

    evaluate_predictions(
        preds_path=preds_path,
        output_dir=OUTPUT_ROOT,
    )

    plot_training_history(
        history_path=history_path,
        output_dir=OUTPUT_ROOT,
    )


