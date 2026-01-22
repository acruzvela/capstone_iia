import logging
from pathlib import Path
import re
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


def leer_documentos(directorio: str):
    docs = []
    path = Path(directorio)
    if not path.exists():
        logger.warning("Directorio %s no existe.", directorio)
        return docs

    for txt_path in sorted(path.glob("*.txt")):
        try:
            texto = txt_path.read_text(encoding="utf-8", errors="ignore")
            if not texto.strip():
                continue

            dt = obtener_fecha_documento(txt_path)
            docs.append((txt_path.name, texto, dt, txt_path))

        except Exception as e:
            logger.warning("Error leyendo %s: %s", txt_path, e)

    logger.info("Leidos %d documentos de %s", len(docs), directorio)
    return docs
    
def obtener_fecha_documento(txt_path: Path):
    """
    1) meta.json -> published/updated/retrieved
    2) filename -> YYYYMMDD_HHMMSS
    3) fallback -> None (no inventar)
    """
    dt = fecha_from_meta(txt_path)
    if dt:
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    dt = fecha_from_name(txt_path.name)
    if dt:
        return dt

    # ❗ No usamos mtime / now: sesga hacia el día de ejecución
    return None

    
    
def fecha_from_name(name: str):
    # 1) YYYYMMDD_HHMMSS_
    m = re.match(r"^(?P<date>\d{8})_(?P<time>\d{6})_", name)
    if m:
        return datetime.strptime(
            m.group("date") + m.group("time"),
            "%Y%m%d%H%M%S"
        ).replace(tzinfo=timezone.utc)

    # 2) YYYYMMDD...
    m2 = re.match(r"^(?P<date>\d{8})_", name)
    if m2:
        d = datetime.strptime(m2.group("date"), "%Y%m%d").date()
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

    return None

def fecha_from_meta(txt_path: Path):
    """
    Lee <archivo>.meta.json y devuelve datetime SOLO si es fecha real de publicación.
    """
    meta_path = txt_path.with_suffix(txt_path.suffix + ".meta.json")  # .txt.meta.json
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    # ✅ SOLO fechas editoriales reales
    for key in ("published", "updated"):
        dt = _parse_iso_datetime(meta.get(key, ""))
        if dt:
            return dt

    # ❌ NO usar retrieved como fecha del documento
    return None


def _parse_iso_datetime(s: str):
    """
    Intenta parsear ISO8601. Acepta 'Z' y offsets.
    Devuelve datetime o None.
    """
    if not s:
        return None
    s = s.strip()
    try:
        # soporta "...Z"
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None