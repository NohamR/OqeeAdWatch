from datetime import datetime
from typing import Dict
from visualizer.utils import format_duration, human_ts, CHANNELS_DATA

def print_stats(channel_id: str, stats: Dict) -> None:
    """Print formatted ad break statistics to the console."""
    if not stats:
        print(f"No ad breaks recorded for channel '{channel_id}'.")
        return

    max_break_duration, max_break_row = stats["max_break"]

    print("\n=== Channel overview ===")
    print(f"Channel ID        : {channel_id}")
    print(f"Total ad breaks   : {stats['count']}")
    print(f"First ad start    : {human_ts(stats['first_start'])}")
    print(f"Latest ad end     : {human_ts(stats['last_end'])}")
    print(f"Total ad duration : {format_duration(stats['total_duration'])}")
    print(f"Mean break length : {format_duration(int(stats['mean_duration']))}")
    print(f"Median break len  : {format_duration(int(stats['median_duration']))}")
    print(
        "Longest break     : "
        f"{format_duration(max_break_duration)} "
        f"({human_ts(max_break_row[1])} -> {human_ts(max_break_row[2])})"
    )

    print("\n=== Per-day breakdown ===")
    print("Date        | Breaks | Total duration | Avg duration")
    print("------------+--------+----------------+-------------")
    for entry in stats["daily_summary"]:
        print(
            f"{entry['date']} | "
            f"{entry['count']:6d} | "
            f"{format_duration(entry['total']).rjust(14)} | "
            f"{format_duration(int(entry['avg'])).rjust(11)}"
        )


def build_overview_text(channel_id: str, stats: Dict, channels_data: Dict = CHANNELS_DATA) -> str:
    """Build a multi-line string with channel overview stats."""
    if not stats:
        return ""
    
    max_break_duration, max_break_row = stats["max_break"]

    channel_name = channel_id
    for ch_id, channel_info in (channels_data or {}).items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]
            break

    lines = [
        f"Channel: {channel_name} ({channel_id})",
        f"Total ad breaks: {stats['count']}",
        f"First ad start: {human_ts(stats['first_start'])}",
        f"Latest ad end: {human_ts(stats['last_end'])}",
        f"Total ad duration: {format_duration(stats['total_duration'])}",
        f"Mean break length: {format_duration(int(stats['mean_duration']))}",
        f"Median break len: {format_duration(int(stats['median_duration']))}",
        f"Longest break: {format_duration(max_break_duration)}",
        f"  ({human_ts(max_break_row[1])} → {human_ts(max_break_row[2])})",
    ]
    return "\n".join(lines)