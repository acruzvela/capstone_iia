"""Transformer market configuration.

Prefer variable de entorno `TRANSFORMER_MARKET_DB` para la ruta a la DB;
en su defecto se usa ../data/ibex35.db relativo al repo.
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Raíz de outputs del módulo Transformer (NO crear a mano)
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "transformer_market"

# DB_PATH: prefer env var TRANSFORMER_MARKET_DB, si no existe usa ../data/ibex35.db
_db_env = os.getenv("TRANSFORMER_MARKET_DB")
if _db_env:
	DB_PATH = Path(_db_env)
else:
	# DB_PATH = PROJECT_ROOT.parent / "data" / "ibex35.db"
	DB_PATH = Path(r"E:/cruz/informatica/sqlite/ibex35.db")

DB_PATH = Path(DB_PATH)

if not DB_PATH.exists():
	logger.warning("transformer_market.config: DB_PATH does not exist: %s", DB_PATH)

