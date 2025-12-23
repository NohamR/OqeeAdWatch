"""Utility functions for the visualizer."""

from datetime import datetime
from pathlib import Path
from typing import Dict
import sys

from utils.scrap import fetch_service_plan

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def get_channel_name(channel_id: str, channels_data: Dict = None) -> str:
    """Get the channel name from channel_id, or return channel_id if not found."""
    if channels_data is None:
        channels_data = CHANNELS_DATA
    channel_name = channel_id
    for ch_id, channel_info in channels_data.items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]
            break
    return channel_name
