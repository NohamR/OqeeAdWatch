from collections import defaultdict
from datetime import datetime, timedelta
import statistics
from typing import Iterable, Sequence, Dict, List

Row = Sequence

# Maximum duration for a single ad break (30 minutes in seconds)
# Breaks longer than this are considered errors and filtered out
MAX_BREAK_DURATION = 30 * 60  # 30 minutes


def _merge_overlapping_breaks(rows: List[Row]) -> List[Row]:
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
    filtered = [row for row in merged if (row[2] - row[1]) <= MAX_BREAK_DURATION]

    return filtered


def compute_stats(rows: Iterable[Row]) -> Dict:
    """Compute overall statistics for ad breaks."""
    rows = list(rows)
    if not rows:
        return {}

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


def compute_hourly_profile(rows: Iterable[Row]) -> Dict:
    """Compute ad statistics grouped by hour of day."""
    rows = list(rows)
    if not rows:
        return {}

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


def compute_heatmap(rows: Iterable[Row]) -> Dict:
    """Compute a heatmap of ad coverage by minute of hour and hour of day."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    heatmap = [[0.0 for _ in range(24)] for _ in range(60)]
    seen_days: set = set()

    for row in merged_rows:
        start_ts, end_ts = row[1], row[2]
        if start_ts >= end_ts:
            continue

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


def compute_weekday_profile(rows: Iterable[Row]) -> Dict:
    """Compute ad stats grouped by day of the week (0=Monday, 6=Sunday)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    weekday_counts = [0] * 7
    weekday_duration = [0] * 7
    weekday_days_seen = [set() for _ in range(7)]

    for row in merged_rows:
        start_dt = datetime.fromtimestamp(row[1])
        weekday = start_dt.weekday()
        duration = row[2] - row[1]
        weekday_counts[weekday] += 1
        weekday_duration[weekday] += duration
        weekday_days_seen[weekday].add(start_dt.date())

    return {
        "counts": weekday_counts,
        "durations": weekday_duration,
        "days_seen": [len(s) for s in weekday_days_seen],
    }


def compute_weekday_hour_counts(rows: Iterable[Row]) -> Dict:
    """Compute a heatmap of ad break counts by weekday (rows) and hour (columns)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    counts = [[0 for _ in range(24)] for _ in range(7)]

    for row in merged_rows:
        start_dt = datetime.fromtimestamp(row[1])
        weekday = start_dt.weekday()
        hour = start_dt.hour
        counts[weekday][hour] += 1

    return {"grid": counts}


def compute_weekday_hour_heatmap(rows: Iterable[Row]) -> Dict:
    """Compute a heatmap of ad coverage by weekday (rows) and hour (columns)."""
    rows = list(rows)
    if not rows:
        return {}

    merged_rows = _merge_overlapping_breaks(rows)

    heatmap = [[0.0 for _ in range(24)] for _ in range(7)]
    weekday_days_seen = [set() for _ in range(7)]

    for row in merged_rows:
        start_ts, end_ts = row[1], row[2]
        if start_ts >= end_ts:
            continue

        current = start_ts
        while current < end_ts:
            dt = datetime.fromtimestamp(current)
            weekday = dt.weekday()
            hour = dt.hour
            weekday_days_seen[weekday].add(dt.date())

            hour_end = current - (current % 3600) + 3600
            overlap = min(end_ts, hour_end) - current
            heatmap[weekday][hour] += overlap
            current = hour_end

    return {
        "grid": heatmap,
        "days_seen": [len(s) for s in weekday_days_seen],
    }