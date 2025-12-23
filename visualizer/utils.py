from datetime import datetime
import sys
from pathlib import Path
from typing import Dict

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.scrap import fetch_service_plan

# Load CHANNELS_DATA once when this module is imported
CHANNELS_DATA: Dict = fetch_service_plan()

def format_duration(seconds: int) -> str:
    """Format a duration in seconds into a human-readable string (e.g., '1h 2m 3s')."""
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def human_ts(ts_value: int) -> str:
    """Convert a Unix timestamp to a human-readable date and time string."""
    return datetime.fromtimestamp(ts_value).strftime("%d/%m/%Y at %H:%M:%S")