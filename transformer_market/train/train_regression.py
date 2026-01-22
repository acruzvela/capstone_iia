# transformer_market/train/train_regression.py

'''
Responsabilidad del archivo

Este script solo hace:
    Cargar el dataset escalado (ibex_w60_p5_scaled.npz)
    Crear splits train / val / test
    Entrenar el modelo con:
        Huber loss
        AdamW
        Early stopping
    Guardar:
        modelo entrenado
        historial de entrenamiento
        predicciones en test (CSV para UI)  

Decisiones de entrenamiento (cerradas)
    Modelo
        TransformerRegressor
        d_model=64, n_layers=2, n_heads=4
    Optimización
        Optimizer: AdamW
        LR inicial: 1e-3
        Weight decay: 1e-4
    Pérdida
        HuberLoss(delta=1.0)    
    Entrenamiento
        Epochs máximas: 50
        Early stopping: patience=7
        Batch size: 32

    Con 477 samples de train, esto es muy razonable.

'''


from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from transformer_market.config import OUTPUT_ROOT
from transformer_market.models.transformer import TransformerRegressor

logger = logging.getLogger(__name__)

# ejecutar python -m transformer_market.train.train_regression

def train_regression(
    npz_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    seed: int = 42,
) -> dict[str, Path]:
    """
    Entrena modelo Transformer para regresion.
    
    Args:
        npz_path: ruta a dataset escalado (ibex_w60_p5_scaled.npz)
        output_dir: directorio para guardar modelo, history, predicciones
        batch_size: tamanio de batch (default: 32)
        max_epochs: epochs maximas (default: 50)
        lr: learning rate (default: 1e-3)
        weight_decay: L2 regularizacion (default: 1e-4)
        patience: early stopping patience (default: 7)
        seed: random seed para reproducibilidad (default: 42)
    
    Returns:
        dict con claves "model", "history", "preds" (rutas a archivos guardados)
    """
    # Reproducibilidad
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Entrenando en: %s", device)

    # ===== Cargar datos =====
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {npz_path}")
    
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Error cargando dataset: {e}") from e

    try:
        X = torch.tensor(data["X"], dtype=torch.float32)
        y = torch.tensor(data["y_reg"], dtype=torch.float32)
        splits = data["split_indices"].item()
        dates = data["dates"]
    except KeyError as e:
        raise ValueError(f"Dataset invalido (falta clave {e})") from e

    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    
    logger.info("Train: %d, Val: %d, Test: %d", len(train_idx), len(val_idx), len(test_idx))

    # ===== Datasets =====
    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds = TensorDataset(X[val_idx], y[val_idx])
    test_ds = TensorDataset(X[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ===== Modelo =====
    model = TransformerRegressor(n_features=X.shape[2])
    model.to(device)

    # ===== Optimización =====
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ===== Early stopping =====
    best_val = float("inf")
    best_state = None
    wait = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    # ===== Entrenamiento =====
    for epoch in range(1, max_epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)

        val_loss /= len(val_ds)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        logger.info(
            "Epoch %03d | Train loss: %.6f | Val loss: %.6f",
            epoch, train_loss, val_loss
        )

        # --- Early stopping ---
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                logger.info("Early stopping activado en epoch %d", epoch)
                break

    # ===== Restaurar mejor modelo =====
    if best_state is None:
        logger.warning("No best_state encontrado; usando modelo actual")
    else:
        model.load_state_dict(best_state)

    # ===== Guardar modelo =====
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "transformer_regressor.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Modelo guardado en: %s", model_path)

    # ===== Guardar history =====
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    history_path = reports_dir / "train_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    logger.info("History guardado en: %s", history_path)

    # ===== Predicciones en test =====
    model.eval()
    preds_test = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            preds_test.append(preds.cpu().numpy())

    preds_test = np.concatenate(preds_test)
    
    # Validacion de NaN
    if np.isnan(preds_test).any():
        logger.warning("Predicciones contienen NaN: %d valores", np.sum(np.isnan(preds_test)))

    preds_df = pd.DataFrame({
        "date": dates[test_idx],
        "y_true": y[test_idx].numpy(),
        "y_pred": preds_test.flatten(),
    })

    preds_dir = output_dir / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    preds_path = preds_dir / "preds_test.csv"
    preds_df.to_csv(preds_path, index=False)
    logger.info("Predicciones test guardadas en: %s", preds_path)

    return {
        "model": model_path,
        "history": history_path,
        "preds": preds_path,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    npz_path = OUTPUT_ROOT / "dataset" / "ibex_w60_p5_scaled.npz"
    train_regression(
        npz_path=npz_path,
        output_dir=OUTPUT_ROOT,
    )

