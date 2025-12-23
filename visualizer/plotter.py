"""Plotting utilities for the ad visualizer."""

from pathlib import Path
from typing import Dict, List, Callable, Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager

from .utils import format_duration, get_channel_name

FPATH = "libs/LibertinusSerif-Regular.otf"
prop = font_manager.FontProperties(fname=FPATH, size=14)

# Register the font file so Matplotlib can find it and use it by default.
try:
    font_manager.fontManager.addfont(FPATH)
    font_name = font_manager.FontProperties(fname=FPATH).get_name()
    if font_name:
        plt.rcParams["font.family"] = font_name
        plt.rcParams["font.size"] = prop.get_size()
except (OSError, ValueError):
    font_name = None


def plot_hourly_profile(
    channel_id: str,
    profile: Dict,
    stats: Dict | None = None,
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
    build_overview_text_func: Callable[[str, Dict], str] = lambda x, y: "",
) -> None:
    """Plot the average ad activity per hour of day."""
    if channels_data is None:
        channels_data = {}
    if not profile or not profile.get("days"):
        print("No data available or not enough distinct days for the hourly plot.")
        return

    hours = list(range(24))
    avg_duration_minutes = [
        (profile["durations"][hour] / profile["days"]) / 60 for hour in hours
    ]
    avg_counts = [profile["counts"][hour] / profile["days"] for hour in hours]

    fig, ax_left = plt.subplots(figsize=(14, 5))
    ax_left.bar(hours, avg_duration_minutes, color="tab:blue", alpha=0.7)
    ax_left.set_xlabel("Hour of day", fontproperties=prop)
    ax_left.set_ylabel(
        "Avg ad duration per day (min)", color="tab:blue", fontproperties=prop
    )
    ax_left.set_xticks(hours)
    ax_left.set_xticklabels([str(h) for h in hours], fontproperties=prop)
    ax_left.set_xlim(-0.5, 23.5)

    ax_right = ax_left.twinx()
    ax_right.plot(hours, avg_counts, color="tab:orange", marker="o")
    ax_right.set_ylabel("Avg number of breaks", color="tab:orange", fontproperties=prop)

    channel_name = get_channel_name(channel_id, channels_data)

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

    if stats:
        overview_text = build_overview_text_func(
            channel_id, stats, channels_data=channels_data
        )
        fig.text(
            0.73,
            0.5,
            overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 1])
    if not save:
        plt.show()

    if save:
        filename = output_dir / f"hourly_profile_{channel_id}.png"
        fig.savefig(filename)
        print(f"Hourly profile saved to {filename}")
    plt.close(fig)


def plot_heatmap(
    channel_id: str,
    heatmap_data: Dict,
    stats: Dict | None = None,
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
    build_overview_text_func: Callable[[str, Dict], str] = lambda x, y: "",
) -> None:
    """Plot a heatmap of ad minute coverage by minute of hour and hour of day."""
    if channels_data is None:
        channels_data = {}
    if not heatmap_data or not heatmap_data.get("days"):
        print("No data available or not enough distinct days for the heatmap plot.")
        return

    days = heatmap_data.get("days", 0)
    normalized = [
        [min(value / (60 * days), 1.0) for value in row] for row in heatmap_data["grid"]
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

    channel_name = get_channel_name(channel_id, channels_data)

    fig.suptitle(
        (
            "Ad minute coverage for channel "
            f"{channel_name} ({channel_id}) across {days} day(s)"
        ),
        fontproperties=prop,
    )

    if stats:
        overview_text = build_overview_text_func(
            channel_id, stats, channels_data=channels_data
        )
        fig.text(
            0.73,
            0.5,
            overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 1])
    if not save:
        plt.show()

    if save:
        filename = output_dir / f"heatmap_{channel_id}.png"
        fig.savefig(filename)
        print(f"Heatmap saved to {filename}")
    plt.close(fig)


def plot_combined(
    channel_id: str,
    profile: Dict,
    heatmap_data: Dict,
    stats: Dict | None = None,
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
    build_overview_text_func: Callable[[str, Dict], str] = lambda x, y: "",
) -> None:
    """Plot both hourly profile and heatmap in a single figure with the overview text box."""
    if channels_data is None:
        channels_data = {}
    if not profile or not profile.get("days"):
        print("No data available for the hourly plot.")
        return
    if not heatmap_data or not heatmap_data.get("days"):
        print("No data available for the heatmap plot.")
        return

    channel_name = get_channel_name(channel_id, channels_data)

    fig, (ax_hourly, ax_heatmap) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Hourly profile (top) ---
    hours = list(range(24))
    avg_duration_minutes = [
        (profile["durations"][hour] / profile["days"]) / 60 for hour in hours
    ]
    avg_counts = [profile["counts"][hour] / profile["days"] for hour in hours]

    ax_hourly.bar(hours, avg_duration_minutes, color="tab:blue", alpha=0.7)
    ax_hourly.set_xlabel("Hour of day", fontproperties=prop)
    ax_hourly.set_ylabel(
        "Avg ad duration per day (min)", color="tab:blue", fontproperties=prop
    )
    ax_hourly.set_xticks(hours)
    ax_hourly.set_xticklabels([str(h) for h in hours], fontproperties=prop)
    ax_hourly.set_xlim(-0.5, 23.5)
    ax_hourly.set_title("Average ad activity by hour", fontproperties=prop)

    ax_hourly_right = ax_hourly.twinx()
    ax_hourly_right.plot(hours, avg_counts, color="tab:orange", marker="o")
    ax_hourly_right.set_ylabel(
        "Avg number of breaks", color="tab:orange", fontproperties=prop
    )

    for t in ax_hourly.get_yticklabels():
        t.set_fontproperties(prop)
    for t in ax_hourly_right.get_yticklabels():
        t.set_fontproperties(prop)

    # --- Heatmap (bottom) ---
    days = heatmap_data.get("days", 0)
    normalized = [
        [min(value / (60 * days), 1.0) for value in row] for row in heatmap_data["grid"]
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

    fig.suptitle(
        f"Ad analysis for {channel_name} ({channel_id}) across {profile['days']} day(s)",
        fontproperties=prop,
        fontsize=16,
    )

    if stats:
        overview_text = build_overview_text_func(
            channel_id, stats, channels_data=channels_data
        )
        fig.text(
            0.73,
            0.5,
            overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 0.96])
    if not save:
        plt.show()

    if save:
        filename = output_dir / f"{channel_id}_combined.png"
        fig.savefig(filename, dpi=300)
        print(f"Combined plot saved to {filename}")
    plt.close(fig)


def plot_weekday_overview(
    all_channels_data: List[Dict],
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
) -> None:
    """
    Plot a weekday overview for all channels.
    Each channel gets:
    - A bar showing number of ads per weekday
    - A horizontal heatmap strip showing ad coverage by weekday x hour
    """
    if channels_data is None:
        channels_data = {}
    if not all_channels_data:
        print("No data available for weekday overview.")
        return

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    num_channels = len(all_channels_data)

    fig, (ax_bars, ax_heatmap) = plt.subplots(
        1, 2, figsize=(18, max(8, num_channels * 0.5))
    )

    channel_names = []
    weekday_counts_all = []
    heatmap_plot_data = []

    for data in all_channels_data:
        channel_id = data["channel_id"]
        channel_name = get_channel_name(channel_id, channels_data)
        channel_names.append(f"{channel_name}")

        weekday_profile = data.get("weekday_profile", {})
        weekday_heatmap = data.get("weekday_heatmap", {})

        counts = weekday_profile.get("counts", [0] * 7)
        days_seen = weekday_profile.get("days_seen", [1] * 7)
        avg_counts = [c / max(d, 1) for c, d in zip(counts, days_seen)]
        weekday_counts_all.append(avg_counts)

        grid = weekday_heatmap.get("grid", [[0] * 24 for _ in range(7)])
        hm_days_seen = weekday_heatmap.get("days_seen", [1] * 7)
        normalized_row = []
        for weekday in range(7):
            for hour in range(24):
                val = grid[weekday][hour] / max(hm_days_seen[weekday], 1) / 3600
                normalized_row.append(min(val, 1.0))
        heatmap_plot_data.append(normalized_row)

    x = range(num_channels)
    bar_width = 0.12
    colors = plt.get_cmap("tab10").colors[:7]

    for i, weekday in enumerate(weekday_names):
        offsets = [xi + (i - 3) * bar_width for xi in x]
        values = [weekday_counts_all[ch][i] for ch in range(num_channels)]
        ax_bars.barh(
            offsets, values, height=bar_width, label=weekday, color=colors[i], alpha=0.8
        )

    ax_bars.set_yticks(list(x))
    ax_bars.set_yticklabels(channel_names, fontproperties=prop)
    ax_bars.set_xlabel("Avg number of ad breaks per day", fontproperties=prop)
    ax_bars.set_title("Ad breaks by day of week", fontproperties=prop)
    ax_bars.legend(title="Day", loc="lower right", fontsize=9)
    ax_bars.invert_yaxis()

    im = ax_heatmap.imshow(
        heatmap_plot_data,
        aspect="auto",
        cmap="Reds",
        vmin=0,
        vmax=0.5,
    )

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

    fig.suptitle(
        "Weekly ad patterns across all channels", fontproperties=prop, fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if not save:
        plt.show()

    if save:
        filename = output_dir / "weekday_overview_all_channels.png"
        fig.savefig(filename, dpi=300)
        print(f"Weekday overview saved to {filename}")
    plt.close(fig)


def plot_weekday_channel(
    channel_id: str,
    weekday_profile: Dict,
    weekday_hour_counts: Dict,
    stats: Dict | None = None,
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
    build_overview_text_func: Callable[[str, Dict], str] = lambda x, y: "",
) -> None:
    """
    Plot a weekday overview for a single channel.
    - Bar chart of ad breaks per weekday
    - Heatmap of ad break counts by weekday x hour (7 rows x 24 columns)
    - Stats text box on the right
    """
    if channels_data is None:
        channels_data = {}
    if not weekday_profile or not weekday_hour_counts:
        print(f"No weekday data available for channel {channel_id}.")
        return

    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    channel_name = get_channel_name(channel_id, channels_data)

    fig, (ax_bars, ax_heatmap) = plt.subplots(2, 1, figsize=(14, 8))

    # --- Top plot: Bar chart for weekday counts ---
    counts = weekday_profile.get("counts", [0] * 7)
    days_seen = weekday_profile.get("days_seen", [1] * 7)
    avg_counts = [c / max(d, 1) for c, d in zip(counts, days_seen)]

    durations = weekday_profile.get("durations", [0] * 7)
    avg_duration_minutes = [d / max(ds, 1) / 60 for d, ds in zip(durations, days_seen)]

    x = range(7)
    bar_width = 0.35

    bars1 = ax_bars.bar(
        [i - bar_width / 2 for i in x],
        avg_counts,
        bar_width,
        label="Avg breaks",
        color="tab:blue",
        alpha=0.7,
    )
    ax_bars.set_ylabel("Avg number of ad breaks", color="tab:blue", fontproperties=prop)
    ax_bars.set_xticks(list(x))
    ax_bars.set_xticklabels(weekday_names, fontproperties=prop)
    ax_bars.set_xlabel("Day of week", fontproperties=prop)
    ax_bars.set_title("Ad breaks by day of week (average per day)", fontproperties=prop)

    ax_bars_right = ax_bars.twinx()
    bars2 = ax_bars_right.bar(
        [i + bar_width / 2 for i in x],
        avg_duration_minutes,
        bar_width,
        label="Avg duration (min)",
        color="tab:orange",
        alpha=0.7,
    )
    ax_bars_right.set_ylabel(
        "Avg ad duration (min)", color="tab:orange", fontproperties=prop
    )

    ax_bars.legend(
        [bars1, bars2], ["Avg breaks", "Avg duration (min)"], loc="upper right"
    )

    for t in ax_bars.get_yticklabels():
        t.set_fontproperties(prop)
    for t in ax_bars_right.get_yticklabels():
        t.set_fontproperties(prop)

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

    fig.suptitle(
        f"Weekly ad patterns for {channel_name} ({channel_id})",
        fontproperties=prop,
        fontsize=16,
    )

    if stats:
        overview_text = build_overview_text_func(
            channel_id, stats, channels_data=channels_data
        )
        fig.text(
            0.73,
            0.5,
            overview_text,
            transform=fig.transFigure,
            fontproperties=prop,
            fontsize=12,
            verticalalignment="center",
            horizontalalignment="left",
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "wheat", "alpha": 0.8},
        )

    fig.tight_layout(rect=[0, 0, 0.72 if stats else 1, 0.96])
    if not save:
        plt.show()

    if save:
        filename = output_dir / f"{channel_id}_weekday.png"
        fig.savefig(filename, dpi=300)
        print(f"Weekday overview saved to {filename}")
    plt.close(fig)


def plot_channel_rankings(
    all_stats: List[Dict],
    save: bool = False,
    output_dir: Path = Path("."),
    channels_data: Optional[Dict] = None,
) -> None:
    """
    Plot rankings of all channels based on:
    - Total number of ads
    - Total ad duration
    - Longest single ad break
    """
    if channels_data is None:
        channels_data = {}
    if not all_stats:
        print("No data available for channel rankings.")
        return

    channels_data_for_plot = []
    for data in all_stats:
        channel_id = data["channel_id"]
        stats = data["stats"]
        if not stats:
            continue

        channel_name = get_channel_name(channel_id, channels_data)

        max_break_duration = stats["max_break"][0] if stats.get("max_break") else 0

        channels_data_for_plot.append(
            {
                "channel_id": channel_id,
                "channel_name": channel_name,
                "total_ads": stats.get("count", 0),
                "total_duration": stats.get("total_duration", 0),
                "longest_break": max_break_duration,
            }
        )

    if not channels_data_for_plot:
        print("No channel data for rankings.")
        return

    fig, axes = plt.subplots(
        1, 3, figsize=(18, max(8, len(channels_data_for_plot) * 0.4))
    )

    rankings = [
        ("total_ads", "Total Number of Ads", "Number of ad breaks", "tab:blue"),
        ("total_duration", "Total Ad Duration", "Duration", "tab:green"),
        ("longest_break", "Longest Single Ad Break", "Duration", "tab:red"),
    ]

    for ax, (metric, title, xlabel, color) in zip(axes, rankings):
        sorted_data = sorted(
            channels_data_for_plot, key=lambda x, m=metric: x[m], reverse=True
        )

        names = [d["channel_name"] for d in sorted_data]
        values = [d[metric] for d in sorted_data]

        if metric in ("total_duration", "longest_break"):
            display_values = values
            labels = [format_duration(int(v)) for v in values]
        else:
            display_values = values
            labels = [str(v) for v in values]

        y_pos = range(len(names))
        bars = ax.barh(y_pos, display_values, color=color, alpha=0.7)

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(names, fontproperties=prop)
        ax.set_xlabel(xlabel, fontproperties=prop)
        ax.set_title(title, fontproperties=prop, fontsize=14)
        ax.invert_yaxis()

        for bar_rect, label in zip(bars, labels):
            width = bar_rect.get_width()
            ax.text(
                width + max(display_values) * 0.01,
                bar_rect.get_y() + bar_rect.get_height() / 2,
                label,
                va="center",
                ha="left",
                fontproperties=prop,
                fontsize=10,
            )

        ax.set_xlim(0, max(display_values) * 1.25)

        for t in ax.get_yticklabels():
            t.set_fontproperties(prop)
        for t in ax.get_xticklabels():
            t.set_fontproperties(prop)

    fig.suptitle("Channel Rankings by Ad Metrics", fontproperties=prop, fontsize=18)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if not save:
        plt.show()

    if save:
        filename = output_dir / "channel_rankings.png"
        fig.savefig(filename, dpi=300)
        print(f"Channel rankings saved to {filename}")
    plt.close(fig)
