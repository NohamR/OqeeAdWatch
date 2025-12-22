"""Channel-level ad break visualizer."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timedelta
import sqlite3
import statistics
from typing import Iterable, Sequence
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FPATH = "libs/LibertinusSerif-Regular.otf"
prop = fm.FontProperties(fname=FPATH, size=14)

# Register the font file so Matplotlib can find it and use it by default.
try:
    fm.fontManager.addfont(FPATH)
    font_name = fm.FontProperties(fname=FPATH).get_name()
    if font_name:
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.size"] = prop.get_size()
except Exception:  # pylint: disable=broad-exception-caught  # pragma: no cover - optional font may be missing
    font_name = None

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.scrap import DB_PATH, get_connection, fetch_service_plan  # pylint: disable=wrong-import-position

Row = Sequence

# Maximum duration for a single ad break (30 minutes in seconds)
# Breaks longer than this are considered errors and filtered out
MAX_BREAK_DURATION = 30 * 60  # 30 minutes


def _merge_overlapping_breaks(rows: list[Row]) -> list[Row]:
    """Merge overlapping ad breaks to avoid double-counting."""
    if not rows:
        return []

    # Sort by start time
    sorted_rows = sorted(rows, key=lambda r: r[1])
    merged = []

    for row in sorted_rows:
        _, start_ts, end_ts, _ = row

        if not merged or merged[-1][2] < start_ts:
            # No overlap with previous break
            merged.append(row)
        else:
            # Overlap detected - merge with previous break
            prev_row = merged[-1]
            new_end = max(prev_row[2], end_ts)
            # Keep the earlier ad_date for consistency
            merged[-1] = (prev_row[0], prev_row[1], new_end, prev_row[3])

    # Filter out breaks longer than MAX_BREAK_DURATION (likely errors)
    filtered = [
        row for row in merged
        if (row[2] - row[1]) <= MAX_BREAK_DURATION
    ]

    return filtered

def _format_duration(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _human_ts(ts_value: int) -> str:
    return datetime.fromtimestamp(ts_value).strftime("%d/%m/%Y at %H:%M:%S")


def _load_rows(
    channel_id: str, start_date: str | None = None, end_date: str | None = None
) -> list[Row]:
    conn = get_connection(DB_PATH)
    try:
        query = """
            SELECT channel_id, start_ts, end_ts, ad_date
            FROM ads WHERE channel_id = ?
        """
        params = [channel_id]

        if start_date:
            query += " AND ad_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND ad_date <= ?"
            params.append(end_date)

        query += " ORDER BY start_ts ASC"

        cursor = conn.execute(query, params)
        return cursor.fetchall()
    except sqlite3.OperationalError as exc:  # pragma: no cover - CLI helper
        raise SystemExit(
            "SQLite query failed. Ensure the collector ran at least once (table 'ads' must exist)."
        ) from exc
    finally:
        conn.close()


def _compute_stats(rows: Iterable[Row]) -> dict:
    rows = list(rows)
    if not rows:
        return {}

    # Merge overlapping breaks to avoid double-counting
    merged_rows = _merge_overlapping_breaks(rows)
    durations = [row[2] - row[1] for row in merged_rows]
    total_duration = sum(durations)

    per_day = defaultdict(list)
    for row, duration in zip(merged_rows, durations):
        per_day[row[3]].append(duration)

    daily_summary = [
        {
            "date": day,
            "count": len(day_durations),
            "total": sum(day_durations),
            "avg": sum(day_durations) / len(day_durations),
        }
        for day, day_durations in sorted(per_day.items())
    ]

    return {
        "count": len(merged_rows),
        "first_start": merged_rows[0][1],
        "last_end": merged_rows[-1][2],
        "total_duration": total_duration,
        "mean_duration": statistics.mean(durations),
        "median_duration": statistics.median(durations),
        "max_break": max(zip(durations, merged_rows), key=lambda item: item[0]),
        "daily_summary": daily_summary,
    }


def _compute_hourly_profile(rows: Iterable[Row]) -> dict:
    rows = list(rows)
    if not rows:
        return {}

    # Merge overlapping breaks to avoid double-counting
    merged_rows = _merge_overlapping_breaks(rows)

    hourly_counts = [0] * 24
    hourly_duration = [0] * 24
    seen_days = set()

    for row in merged_rows:
        start_dt = datetime.fromtimestamp(row[1])
        seen_days.add(start_dt.date())
        hour = start_dt.hour
        duration = row[2] - row[1]
        hourly_counts[hour] += 1
        hourly_duration[hour] += duration

    return {
        "days": len(seen_days),
        "counts": hourly_counts,
        "durations": hourly_duration,
    }


def _compute_heatmap(rows: Iterable[Row]) -> dict:
    rows = list(rows)
    if not rows:
        return {}

    # Merge overlapping breaks to avoid double-counting
    merged_rows = _merge_overlapping_breaks(rows)

    heatmap = [[0.0 for _ in range(24)] for _ in range(60)]
    seen_days: set = set()

    for row in merged_rows:
        start_ts, end_ts = row[1], row[2]
        if start_ts >= end_ts:
            continue

        # Track every day touched by this break for normalization later.
        day_cursor = datetime.fromtimestamp(start_ts).date()
        last_day = datetime.fromtimestamp(end_ts - 1).date()
        while day_cursor <= last_day:
            seen_days.add(day_cursor)
            day_cursor += timedelta(days=1)

        bucket_start = (start_ts // 60) * 60
        bucket_end = ((end_ts + 59) // 60) * 60

        current = bucket_start
        while current < bucket_end:
            next_bucket = current + 60
            overlap = max(0, min(end_ts, next_bucket) - max(start_ts, current))
            if overlap > 0:
                dt = datetime.fromtimestamp(current)
                heatmap[dt.minute][dt.hour] += overlap
            current = next_bucket

    return {"grid": heatmap, "days": len(seen_days)}


def _compute_weekday_profile(rows: Iterable[Row]) -> dict:
    """Compute ad stats grouped by day of the week (0=Monday, 6=Sunday)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    # Initialize counters for each day of week
    weekday_counts = [0] * 7  # Number of ad breaks
    weekday_duration = [0] * 7  # Total duration in seconds
    weekday_days_seen = [set() for _ in range(7)]  # Unique dates per weekday

    for row in merged_rows:
        start_dt = datetime.fromtimestamp(row[1])
        weekday = start_dt.weekday()  # 0=Monday, 6=Sunday
        duration = row[2] - row[1]
        weekday_counts[weekday] += 1
        weekday_duration[weekday] += duration
        weekday_days_seen[weekday].add(start_dt.date())

    return {
        "counts": weekday_counts,
        "durations": weekday_duration,
        "days_seen": [len(s) for s in weekday_days_seen],
    }


def _compute_weekday_hour_counts(rows: Iterable[Row]) -> dict:
    """Compute a heatmap of ad break counts by weekday (rows) and hour (columns)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    # 7 weekdays x 24 hours - store count of ad breaks
    counts = [[0 for _ in range(24)] for _ in range(7)]

    for row in merged_rows:
        start_dt = datetime.fromtimestamp(row[1])
        weekday = start_dt.weekday()
        hour = start_dt.hour
        counts[weekday][hour] += 1

    return {"grid": counts}


def _compute_weekday_hour_heatmap(rows: Iterable[Row]) -> dict:
    """Compute a heatmap of ad coverage by weekday (rows) and hour (columns)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    # 7 weekdays x 24 hours - store total seconds of ads
    heatmap = [[0.0 for _ in range(24)] for _ in range(7)]
    weekday_days_seen = [set() for _ in range(7)]

    for row in merged_rows:
        start_ts, end_ts = row[1], row[2]
        if start_ts >= end_ts:
            continue

        # Iterate through each hour bucket touched by this ad break
        current = start_ts
        while current < end_ts:
            dt = datetime.fromtimestamp(current)
            weekday = dt.weekday()
            hour = dt.hour
            weekday_days_seen[weekday].add(dt.date())

            # Calculate overlap with this hour bucket
            hour_end = current - (current % 3600) + 3600  # End of current hour
            overlap = min(end_ts, hour_end) - current
            heatmap[weekday][hour] += overlap
            current = hour_end

    return {
        "grid": heatmap,
        "days_seen": [len(s) for s in weekday_days_seen],
    }


def _print_stats(channel_id: str, stats: dict) -> None:
    if not stats:
        print(f"No ad breaks recorded for channel '{channel_id}'.")
        return

    duration_fmt = _format_duration
    max_break_duration, max_break_row = stats["max_break"]

    print("\n=== Channel overview ===")
    print(f"Channel ID        : {channel_id}")
    print(f"Total ad breaks   : {stats['count']}")
    print(f"First ad start    : {_human_ts(stats['first_start'])}")
    print(f"Latest ad end     : {_human_ts(stats['last_end'])}")
    print(f"Total ad duration : {duration_fmt(stats['total_duration'])}")
    print(f"Mean break length : {duration_fmt(int(stats['mean_duration']))}")
    print(f"Median break len  : {duration_fmt(int(stats['median_duration']))}")
    print(
        "Longest break     : "
        f"{duration_fmt(max_break_duration)} "
        f"({_human_ts(max_break_row[1])} -> {_human_ts(max_break_row[2])})"
    )

    print("\n=== Per-day breakdown ===")
    print("Date        | Breaks | Total duration | Avg duration")
    print("------------+--------+----------------+-------------")
    for entry in stats["daily_summary"]:
        print(
            f"{entry['date']} | "
            f"{entry['count']:6d} | "
            f"{duration_fmt(entry['total']).rjust(14)} | "
            f"{duration_fmt(int(entry['avg'])).rjust(11)}"
        )


def _build_overview_text(channel_id: str, stats: dict) -> str:
    """Build a multi-line string with channel overview stats."""
    if not stats:
        return ""
    duration_fmt = _format_duration
    max_break_duration, max_break_row = stats["max_break"]

    channel_name = channel_id
    for ch_id, channel_info in (CHANNELS_DATA or {}).items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]
            break

    lines = [
        f"Channel: {channel_name} ({channel_id})",
        f"Total ad breaks: {stats['count']}",
        f"First ad start: {_human_ts(stats['first_start'])}",
        f"Latest ad end: {_human_ts(stats['last_end'])}",
        f"Total ad duration: {duration_fmt(stats['total_duration'])}",
        f"Mean break length: {duration_fmt(int(stats['mean_duration']))}",
        f"Median break len: {duration_fmt(int(stats['median_duration']))}",
        f"Longest break: {duration_fmt(max_break_duration)}",
        f"  ({_human_ts(max_break_row[1])} → {_human_ts(max_break_row[2])})",
    ]
    return "\n".join(lines)


def _plot_hourly_profile(channel_id: str, profile: dict, stats: dict | None = None, save=False) -> None:
    if not profile:
        print("No data available for the hourly plot.")
        return
    if not profile["days"]:
        print("Not enough distinct days to build an hourly average plot.")
        return

    hours = list(range(24))
    avg_duration_minutes = [
        (profile["durations"][hour] / profile["days"]) / 60 for hour in hours
    ]
    avg_counts = [profile["counts"][hour] / profile["days"] for hour in hours]

    fig, ax_left = plt.subplots(figsize=(14, 5))
    ax_left.bar(hours, avg_duration_minutes, color="tab:blue", alpha=0.7)
    ax_left.set_xlabel("Hour of day", fontproperties=prop)
    ax_left.set_ylabel("Avg ad duration per day (min)", color="tab:blue", fontproperties=prop)
    ax_left.set_xticks(hours)
    ax_left.set_xticklabels([str(h) for h in hours], fontproperties=prop)
    ax_left.set_xlim(-0.5, 23.5)

    ax_right = ax_left.twinx()
    ax_right.plot(hours, avg_counts, color="tab:orange", marker="o")
    ax_right.set_ylabel("Avg number of breaks", color="tab:orange", fontproperties=prop)

    channel_name = channel_id
    for ch_id, channel_info in (CHANNELS_DATA or {}).items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]

    for t in ax_left.get_yticklabels():
        t.set_fontproperties(prop)
    for t in ax_right.get_yticklabels():
        t.set_fontproperties(prop)

    fig.suptitle(
        (
            "Average ad activity for channel "
            f"{channel_name} ({channel_id}) across {profile['days']} day(s)"
        ),
        fontproperties=prop,
    )

    # Add channel overview text box if stats provided
    if stats:
        overview_text = _build_overview_text(channel_id, stats)
        fig.text(
            0.73, 0.5, overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 1])
    plt.show()

    if save:
        filename = f"visualizer/hourly_profile_{channel_id}.png"
        fig.savefig(filename)
        print(f"Hourly profile saved to {filename}")


def _plot_heatmap(channel_id: str, heatmap: dict, stats: dict | None = None, save=False) -> None:
    if not heatmap:
        print("No data available for the heatmap plot.")
        return
    days = heatmap.get("days", 0)
    if not days:
        print("Not enough distinct days to build a heatmap.")
        return

    normalized = [
        [min(value / (60 * days), 1.0) for value in row]
        for row in heatmap["grid"]
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        normalized,
        origin="lower",
        aspect="auto",
        cmap="Reds",
        extent=[0, 24, 0, 60],
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Hour of day", fontproperties=prop)
    ax.set_ylabel("Minute within hour", fontproperties=prop)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([str(x) for x in range(0, 25, 2)], fontproperties=prop)
    ax.set_yticks(range(0, 61, 10))
    ax.set_yticklabels([str(y) for y in range(0, 61, 10)], fontproperties=prop)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share of minute spent in ads per day", fontproperties=prop)

    channel_name = channel_id
    for ch_id, channel_info in CHANNELS_DATA.items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]

    fig.suptitle(
        (
            "Ad minute coverage for channel "
            f"{channel_name} ({channel_id}) across {days} day(s)"
        ),
        fontproperties=prop,
    )

    # Add channel overview text box if stats provided
    if stats:
        overview_text = _build_overview_text(channel_id, stats)
        fig.text(
            0.73, 0.5, overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 1])
    plt.show()

    if save:
        filename = f"visualizer/heatmap_{channel_id}.png"
        fig.savefig(filename)
        print(f"Heatmap saved to {filename}")


def _plot_combined(channel_id: str, profile: dict, heatmap: dict, stats: dict | None = None, save=False) -> None:
    """Plot both hourly profile and heatmap in a single figure with the overview text box."""
    if not profile or not profile.get("days"):
        print("No data available for the hourly plot.")
        return
    if not heatmap or not heatmap.get("days"):
        print("No data available for the heatmap plot.")
        return

    channel_name = channel_id
    for ch_id, channel_info in (CHANNELS_DATA or {}).items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]
            break

    # Create figure with 2 rows
    fig, (ax_hourly, ax_heatmap) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Hourly profile (top) ---
    hours = list(range(24))
    avg_duration_minutes = [
        (profile["durations"][hour] / profile["days"]) / 60 for hour in hours
    ]
    avg_counts = [profile["counts"][hour] / profile["days"] for hour in hours]

    ax_hourly.bar(hours, avg_duration_minutes, color="tab:blue", alpha=0.7)
    ax_hourly.set_xlabel("Hour of day", fontproperties=prop)
    ax_hourly.set_ylabel("Avg ad duration per day (min)", color="tab:blue", fontproperties=prop)
    ax_hourly.set_xticks(hours)
    ax_hourly.set_xticklabels([str(h) for h in hours], fontproperties=prop)
    ax_hourly.set_xlim(-0.5, 23.5)
    ax_hourly.set_title("Average ad activity by hour", fontproperties=prop)

    ax_hourly_right = ax_hourly.twinx()
    ax_hourly_right.plot(hours, avg_counts, color="tab:orange", marker="o")
    ax_hourly_right.set_ylabel("Avg number of breaks", color="tab:orange", fontproperties=prop)

    for t in ax_hourly.get_yticklabels():
        t.set_fontproperties(prop)
    for t in ax_hourly_right.get_yticklabels():
        t.set_fontproperties(prop)

    # --- Heatmap (bottom) ---
    days = heatmap.get("days", 0)
    normalized = [
        [min(value / (60 * days), 1.0) for value in row]
        for row in heatmap["grid"]
    ]

    im = ax_heatmap.imshow(
        normalized,
        origin="lower",
        aspect="auto",
        cmap="Reds",
        extent=[0, 24, 0, 60],
        vmin=0,
        vmax=1,
    )
    ax_heatmap.set_xlabel("Hour of day", fontproperties=prop)
    ax_heatmap.set_ylabel("Minute within hour", fontproperties=prop)
    ax_heatmap.set_xticks(range(0, 25, 2))
    ax_heatmap.set_xticklabels([str(x) for x in range(0, 25, 2)], fontproperties=prop)
    ax_heatmap.set_yticks(range(0, 61, 10))
    ax_heatmap.set_yticklabels([str(y) for y in range(0, 61, 10)], fontproperties=prop)
    ax_heatmap.set_title("Ad minute coverage heatmap", fontproperties=prop)

    cbar = fig.colorbar(im, ax=ax_heatmap)
    cbar.set_label("Share of minute spent in ads per day", fontproperties=prop)

    # Main title
    fig.suptitle(
        f"Ad analysis for {channel_name} ({channel_id}) across {profile['days']} day(s)",
        fontproperties=prop,
        fontsize=16,
    )

    # Add channel overview text box if stats provided
    if stats:
        overview_text = _build_overview_text(channel_id, stats)
        fig.text(
            0.73, 0.5, overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 0.96])
    plt.show()

    if save:
        filename = f"visualizer/{channel_id}_combined.png"
        fig.savefig(filename, dpi=300)
        print(f"Combined plot saved to {filename}")


def _plot_weekday_overview(all_channels_data: list[dict], save=False) -> None:
    """
    Plot a weekday overview for all channels.
    
    Each channel gets:
    - A bar showing number of ads per weekday
    - A horizontal heatmap strip showing ad coverage by weekday x hour
    """
    if not all_channels_data:
        print("No data available for weekday overview.")
        return

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    num_channels = len(all_channels_data)

    # Create figure with 2 subplots side by side
    fig, (ax_bars, ax_heatmap) = plt.subplots(1, 2, figsize=(18, max(8, num_channels * 0.5)))

    # Prepare data for plotting
    channel_names = []
    weekday_counts_all = []
    heatmap_data = []

    for data in all_channels_data:
        channel_id = data["channel_id"]
        channel_name = channel_id
        for ch_id, channel_info in (CHANNELS_DATA or {}).items():
            if ch_id == channel_id:
                channel_name = channel_info["name"]
                break
        channel_names.append(f"{channel_name}")

        weekday_profile = data.get("weekday_profile", {})
        weekday_heatmap = data.get("weekday_heatmap", {})

        # Get average counts per weekday
        counts = weekday_profile.get("counts", [0] * 7)
        days_seen = weekday_profile.get("days_seen", [1] * 7)
        avg_counts = [c / max(d, 1) for c, d in zip(counts, days_seen)]
        weekday_counts_all.append(avg_counts)

        # Get heatmap grid (7 weekdays x 24 hours) and normalize
        grid = weekday_heatmap.get("grid", [[0] * 24 for _ in range(7)])
        hm_days_seen = weekday_heatmap.get("days_seen", [1] * 7)
        # Normalize: average seconds per hour per day, then convert to fraction of hour
        normalized_row = []
        for weekday in range(7):
            for hour in range(24):
                val = grid[weekday][hour] / max(hm_days_seen[weekday], 1) / 3600  # Fraction of hour
                normalized_row.append(min(val, 1.0))
        heatmap_data.append(normalized_row)

    # --- Left plot: Grouped bar chart for weekday counts ---
    x = range(num_channels)
    bar_width = 0.12
    colors = plt.cm.tab10(range(7))

    for i, weekday in enumerate(weekday_names):
        offsets = [xi + (i - 3) * bar_width for xi in x]
        values = [weekday_counts_all[ch][i] for ch in range(num_channels)]
        ax_bars.barh(offsets, values, height=bar_width, label=weekday, color=colors[i], alpha=0.8)

    ax_bars.set_yticks(list(x))
    ax_bars.set_yticklabels(channel_names, fontproperties=prop)
    ax_bars.set_xlabel("Avg number of ad breaks per day", fontproperties=prop)
    ax_bars.set_title("Ad breaks by day of week", fontproperties=prop)
    ax_bars.legend(title="Day", loc="lower right", fontsize=9)
    ax_bars.invert_yaxis()

    # --- Right plot: Heatmap with 7 days x 24 hours per channel as horizontal strips ---
    # Each channel is a row, with 7*24=168 columns (Mon 0h, Mon 1h, ..., Sun 23h)
    heatmap_array = heatmap_data

    im = ax_heatmap.imshow(
        heatmap_array,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=0.5,  # Cap at 50% of hour in ads for visibility
    )

    # X-axis: mark each day boundary
    ax_heatmap.set_xticks([i * 24 + 12 for i in range(7)])
    ax_heatmap.set_xticklabels(weekday_names, fontproperties=prop)
    for i in range(1, 7):
        ax_heatmap.axvline(x=i * 24 - 0.5, color="white", linewidth=1)

    ax_heatmap.set_yticks(list(range(num_channels)))
    ax_heatmap.set_yticklabels(channel_names, fontproperties=prop)
    ax_heatmap.set_xlabel("Day of week (each day spans 24 hours)", fontproperties=prop)
    ax_heatmap.set_title("Ad coverage heatmap by weekday & hour", fontproperties=prop)

    cbar = fig.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label("Fraction of hour in ads (avg per day)", fontproperties=prop)

    fig.suptitle("Weekly ad patterns across all channels", fontproperties=prop, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if save:
        filename = "visualizer/weekday_overview_all_channels.png"
        fig.savefig(filename, dpi=300)
        print(f"Weekday overview saved to {filename}")


def _plot_weekday_channel(channel_id: str, weekday_profile: dict, weekday_hour_counts: dict, stats: dict | None = None, save=False) -> None:
    """
    Plot a weekday overview for a single channel.
    
    Shows:
    - Bar chart of ad breaks per weekday
    - Heatmap of ad break counts by weekday x hour (7 rows x 24 columns)
    - Stats text box on the right
    """
    if not weekday_profile or not weekday_hour_counts:
        print(f"No weekday data available for channel {channel_id}.")
        return

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    channel_name = channel_id
    for ch_id, channel_info in (CHANNELS_DATA or {}).items():
        if ch_id == channel_id:
            channel_name = channel_info["name"]
            break

    # Create figure with 2 subplots stacked vertically
    fig, (ax_bars, ax_heatmap) = plt.subplots(2, 1, figsize=(14, 8))

    # --- Top plot: Bar chart for weekday counts ---
    counts = weekday_profile.get("counts", [0] * 7)
    days_seen = weekday_profile.get("days_seen", [1] * 7)
    avg_counts = [c / max(d, 1) for c, d in zip(counts, days_seen)]

    durations = weekday_profile.get("durations", [0] * 7)
    avg_duration_minutes = [d / max(ds, 1) / 60 for d, ds in zip(durations, days_seen)]

    x = range(7)
    bar_width = 0.35

    bars1 = ax_bars.bar([i - bar_width/2 for i in x], avg_counts, bar_width, label="Avg breaks", color="tab:blue", alpha=0.7)
    ax_bars.set_ylabel("Avg number of ad breaks", color="tab:blue", fontproperties=prop)
    ax_bars.set_xticks(list(x))
    ax_bars.set_xticklabels(weekday_names, fontproperties=prop)
    ax_bars.set_xlabel("Day of week", fontproperties=prop)
    ax_bars.set_title("Ad breaks by day of week (average per day)", fontproperties=prop)

    ax_bars_right = ax_bars.twinx()
    bars2 = ax_bars_right.bar([i + bar_width/2 for i in x], avg_duration_minutes, bar_width, label="Avg duration (min)", color="tab:orange", alpha=0.7)
    ax_bars_right.set_ylabel("Avg ad duration (min)", color="tab:orange", fontproperties=prop)

    # Combined legend
    ax_bars.legend([bars1, bars2], ["Avg breaks", "Avg duration (min)"], loc="upper right")

    for t in ax_bars.get_yticklabels():
        t.set_fontproperties(prop)
    for t in ax_bars_right.get_yticklabels():
        t.set_fontproperties(prop)

    # --- Bottom plot: Heatmap (7 weekdays x 24 hours) - total break counts ---
    grid = weekday_hour_counts.get("grid", [[0] * 24 for _ in range(7)])

    im = ax_heatmap.imshow(
        grid,
        aspect="auto",
        cmap="Reds",
        origin="upper",
    )

    ax_heatmap.set_xticks(range(0, 24, 2))
    ax_heatmap.set_xticklabels([str(h) for h in range(0, 24, 2)], fontproperties=prop)
    ax_heatmap.set_yticks(range(7))
    ax_heatmap.set_yticklabels(weekday_names, fontproperties=prop)
    ax_heatmap.set_xlabel("Hour of day", fontproperties=prop)
    ax_heatmap.set_ylabel("Day of week", fontproperties=prop)
    ax_heatmap.set_title("Total ad breaks by weekday & hour", fontproperties=prop)

    cbar = fig.colorbar(im, ax=ax_heatmap, shrink=0.8)
    cbar.set_label("Number of ad breaks", fontproperties=prop)

    # Main title
    fig.suptitle(
        f"Weekly ad patterns for {channel_name} ({channel_id})",
        fontproperties=prop,
        fontsize=16,
    )

    # Add channel overview text box if stats provided
    if stats:
        overview_text = _build_overview_text(channel_id, stats)
        fig.text(
            0.73, 0.5, overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 0.96])
    plt.show()

    if save:
        filename = f"visualizer/{channel_id}_weekday.png"
        fig.savefig(filename, dpi=300)
        print(f"Weekday overview saved to {filename}")


def list_channels() -> list[str]:
    """List all channel IDs present in the database."""
    conn = get_connection(DB_PATH)
    try:
        cursor = conn.execute("SELECT DISTINCT channel_id FROM ads ORDER BY channel_id ASC")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def _plot_channel_rankings(all_stats: list[dict], save=False) -> None:
    """
    Plot rankings of all channels based on:
    - Total number of ads
    - Total ad duration
    - Longest single ad break
    """
    if not all_stats:
        print("No data available for channel rankings.")
        return

    # Extract data for each ranking metric
    channels_data = []
    for data in all_stats:
        channel_id = data["channel_id"]
        stats = data["stats"]
        if not stats:
            continue

        channel_name = channel_id
        for ch_id, channel_info in (CHANNELS_DATA or {}).items():
            if ch_id == channel_id:
                channel_name = channel_info["name"]
                break

        max_break_duration = stats["max_break"][0] if stats.get("max_break") else 0

        channels_data.append({
            "channel_id": channel_id,
            "channel_name": channel_name,
            "total_ads": stats.get("count", 0),
            "total_duration": stats.get("total_duration", 0),
            "longest_break": max_break_duration,
        })

    if not channels_data:
        print("No channel data for rankings.")
        return

    # Create figure with 3 subplots (one for each ranking)
    fig, axes = plt.subplots(1, 3, figsize=(18, max(8, len(channels_data) * 0.4)))

    rankings = [
        ("total_ads", "Total Number of Ads", "Number of ad breaks", "tab:blue"),
        ("total_duration", "Total Ad Duration", "Duration", "tab:green"),
        ("longest_break", "Longest Single Ad Break", "Duration", "tab:red"),
    ]

    for ax, (metric, title, xlabel, color) in zip(axes, rankings):
        # Sort by the metric (descending)
        sorted_data = sorted(channels_data, key=lambda x: x[metric], reverse=True)

        names = [d["channel_name"] for d in sorted_data]
        values = [d[metric] for d in sorted_data]

        # Format values for duration metrics
        if metric in ("total_duration", "longest_break"):
            display_values = values
            # Create labels with formatted duration
            labels = [_format_duration(int(v)) for v in values]
        else:
            display_values = values
            labels = [str(v) for v in values]

        y_pos = range(len(names))
        bars = ax.barh(y_pos, display_values, color=color, alpha=0.7)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontproperties=prop)
        ax.set_xlabel(xlabel, fontproperties=prop)
        ax.set_title(title, fontproperties=prop, fontsize=14)
        ax.invert_yaxis()  # Highest at top

        # Add value labels on bars
        for i, (bar, label) in enumerate(zip(bars, labels)):
            width = bar.get_width()
            ax.text(
                width + max(display_values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontproperties=prop,
                fontsize=10,
            )

        # Extend x-axis to make room for labels
        ax.set_xlim(0, max(display_values) * 1.25)

        for t in ax.get_yticklabels():
            t.set_fontproperties(prop)
        for t in ax.get_xticklabels():
            t.set_fontproperties(prop)

    fig.suptitle("Channel Rankings by Ad Metrics", fontproperties=prop, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    if save:
        filename = "visualizer/channel_rankings.png"
        fig.savefig(filename, dpi=300)
        print(f"Channel rankings saved to {filename}")


def process_all_channels(start_date, end_date) -> None:
    """Process all channels in the database and generate visualizations."""
    # clear visualizer output directory

    output_dir = Path("visualizer")
    output_dir.mkdir(exist_ok=True)
    for file in output_dir.glob("*.png"):
        file.unlink()
    channel_ids = list_channels()
    
    # Collect data for all channels (for the weekday overview plot)
    all_channels_data = []
    # Collect stats for all channels (for the rankings plot)
    all_stats = []
    
    for channel_id in channel_ids:
        print(f"Processing channel {channel_id}...")
        rows = _load_rows(channel_id, start_date, end_date)
        stats = _compute_stats(rows)
        _print_stats(channel_id, stats)

        hourly_profile = _compute_hourly_profile(rows)
        heatmap = _compute_heatmap(rows)
        _plot_combined(channel_id, hourly_profile, heatmap, stats=stats, save=True)
        
        # Compute weekday data for the overview plot
        weekday_profile = _compute_weekday_profile(rows)
        weekday_heatmap = _compute_weekday_hour_heatmap(rows)
        weekday_hour_counts = _compute_weekday_hour_counts(rows)
        
        # Generate individual weekday overview for this channel
        _plot_weekday_channel(channel_id, weekday_profile, weekday_hour_counts, stats=stats, save=True)
        
        all_channels_data.append({
            "channel_id": channel_id,
            "weekday_profile": weekday_profile,
            "weekday_heatmap": weekday_heatmap,
        })
        
        # Collect stats for rankings
        all_stats.append({
            "channel_id": channel_id,
            "stats": stats,
        })
    
    # Generate the weekday overview plot for all channels
    _plot_weekday_overview(all_channels_data, save=True)
    
    # Generate the channel rankings plot
    _plot_channel_rankings(all_stats, save=True)


def main() -> None:
    """CLI entrypoint for visualizing ad breaks."""
    parser = argparse.ArgumentParser(
        description="Inspect ad breaks for channels from the local database.",
    )
    parser.add_argument(
        "channel_id",
        nargs="?",
        default="all",
        help="Channel identifier to inspect, or 'all' to process all channels (default: all)",
    )
    parser.add_argument(
        "--start-date",
        help="Start date for filtering (YYYY-MM-DD format, inclusive)",
    )
    parser.add_argument(
        "--end-date",
        help="End date for filtering (YYYY-MM-DD format, inclusive)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the matplotlib chart and only print textual stats.",
    )
    args = parser.parse_args()

    if args.channel_id.lower() == "all":
        # Process all channels
        process_all_channels(args.start_date, args.end_date)
    else:
        # Process single channel
        rows = _load_rows(args.channel_id, args.start_date, args.end_date)
        stats = _compute_stats(rows)
        _print_stats(args.channel_id, stats)

        if not args.no_plot:
            hourly_profile = _compute_hourly_profile(rows)
            _plot_hourly_profile(args.channel_id, hourly_profile, stats=stats)
            heatmap = _compute_heatmap(rows)
            _plot_heatmap(args.channel_id, heatmap, stats=stats)


if __name__ == "__main__":
    CHANNELS_DATA = fetch_service_plan()
    main()