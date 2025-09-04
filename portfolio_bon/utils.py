from pathlib import Path
from datetime import datetime

try:
    from zoneinfo import ZoneInfo  # py>=3.9

    _TZ = ZoneInfo("America/New_York")
except Exception:
    _TZ = None


def timestamp(fmt="%Y%m%d_%H%M%S"):
    now = datetime.now(_TZ) if _TZ else datetime.now()
    return now.strftime(fmt)


def timestamped_path(path: str, sep: str = "_", ts: str | None = None) -> str:
    """Insert timestamp before extension: plot.png -> plot_YYYYmmdd_HHMMSS.png"""
    p = Path(path)
    ts = ts or timestamp()
    return str(p.with_name(f"{p.stem}{sep}{ts}{p.suffix}"))
