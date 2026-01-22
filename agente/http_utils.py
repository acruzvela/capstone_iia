import requests
from requests.adapters import HTTPAdapter
try:
    # Prefer direct urllib3 import (recommended)
    from urllib3.util.retry import Retry
except Exception as e:
    # Provide a clear error if urllib3 isn't available
    raise ImportError(
        "urllib3 is required for HTTP retry support. Install urllib3 or a recent requests package."
    ) from e


def get_session(retries: int = 3, backoff_factor: float = 0.3, status_forcelist=(429, 500, 502, 503, 504), user_agent: str | None = None) -> requests.Session:
    """Return a requests.Session configured with retries and a User-Agent header.

    - retries: number of total retries (on connection and on specific status codes)
    - backoff_factor: sleep multiplier between attempts
    - status_forcelist: HTTP status codes that should trigger a retry
    - user_agent: optional User-Agent string; if None, a sensible default is used
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=("HEAD", "GET", "OPTIONS", "POST")
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    ua = user_agent or "CAPSTONE/1.0 (+https://example.com)"
    session.headers.update({"User-Agent": ua, "Accept": "*/*"})
    return session
