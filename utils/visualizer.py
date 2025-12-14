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

    return merged

def _format_duration(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _human_ts(ts_value: int) -> str:
    return datetime.fromtimestamp(ts_value).strftime("%Y-%m-%d %H:%M:%S")


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


def _plot_hourly_profile(channel_id: str, profile: dict, save=False) -> None:
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

    fig, ax_left = plt.subplots(figsize=(10, 5))
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
    fig.tight_layout()
    plt.show()

    if save:
        filename = f"visualizer/hourly_profile_{channel_id}.png"
        fig.savefig(filename)
        print(f"Hourly profile saved to {filename}")


def _plot_heatmap(channel_id: str, heatmap: dict, save=False) -> None:
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

    fig, ax = plt.subplots(figsize=(10, 5))
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
    fig.tight_layout()
    plt.show()

    if save:
        filename = f"visualizer/heatmap_{channel_id}.png"
        fig.savefig(filename)
        print(f"Heatmap saved to {filename}")


def main() -> None:
    """CLI entrypoint for visualizing ad breaks."""
    parser = argparse.ArgumentParser(
        description="Inspect ad breaks for a single channel from the local database.",
    )
    parser.add_argument("channel_id", help="Exact channel identifier to inspect")
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

    rows = _load_rows(args.channel_id, args.start_date, args.end_date)
    stats = _compute_stats(rows)
    _print_stats(args.channel_id, stats)

    if not args.no_plot:
        hourly_profile = _compute_hourly_profile(rows)
        _plot_hourly_profile(args.channel_id, hourly_profile)
        heatmap = _compute_heatmap(rows)
        _plot_heatmap(args.channel_id, heatmap)


def list_channels() -> list[str]:
    """List all channel IDs present in the database."""
    conn = get_connection(DB_PATH)
    try:
        cursor = conn.execute("SELECT DISTINCT channel_id FROM ads ORDER BY channel_id ASC")
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def process_all_channels() -> None:
    """Process all channels in the database and generate visualizations."""
    # clear visualizer output directory
    output_dir = Path("visualizer")
    output_dir.mkdir(exist_ok=True)
    for file in output_dir.glob("*.png"):
        file.unlink()
    channel_ids = list_channels()
    for channel_id in channel_ids:
        print(f"Processing channel {channel_id}...")
        rows = _load_rows(channel_id)
        stats = _compute_stats(rows)
        _print_stats(channel_id, stats)

        hourly_profile = _compute_hourly_profile(rows)
        _plot_hourly_profile(channel_id, hourly_profile, save=True)
        heatmap = _compute_heatmap(rows)
        _plot_heatmap(channel_id, heatmap, save=True)


if __name__ == "__main__":
    CHANNELS_DATA = fetch_service_plan()
    # main()
    process_all_channels()