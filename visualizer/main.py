"""Channel-level ad break visualizer."""

import argparse
import sys
from pathlib import Path

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualizer.data_loader import load_ads_data, list_channels
from visualizer.stats_computer import (
    compute_stats,
    compute_hourly_profile,
    compute_heatmap,
    compute_weekday_profile,
    compute_weekday_hour_counts,
    compute_weekday_hour_heatmap,
)
from visualizer.plotter import (
    plot_hourly_profile,
    plot_heatmap,
    plot_combined,
    plot_weekday_overview,
    plot_weekday_channel,
    plot_channel_rankings,
)
from visualizer.text_output import print_stats, build_overview_text
from visualizer.utils import CHANNELS_DATA

def process_all_channels(start_date, end_date) -> None:
    """Process all channels in the database and generate visualizations."""
    output_dir = Path("visualizer_output")
    output_dir.mkdir(exist_ok=True)
    for file in output_dir.glob("*.png"):
        file.unlink()
    channel_ids = list_channels()

    all_channels_plot_data = [] # Data for combined weekday plots
    all_channels_ranking_data = [] # Data for channel rankings

    for channel_id in channel_ids:
        print(f"Processing channel {channel_id}...")
        rows = load_ads_data(channel_id, start_date, end_date)
        stats = compute_stats(rows)
        print_stats(channel_id, stats)

        hourly_profile = compute_hourly_profile(rows)
        heatmap = compute_heatmap(rows)
        plot_combined(channel_id, hourly_profile, heatmap, stats=stats, save=True, output_dir=output_dir, channels_data=CHANNELS_DATA, build_overview_text_func=build_overview_text)

        weekday_profile = compute_weekday_profile(rows)
        weekday_heatmap = compute_weekday_hour_heatmap(rows)
        weekday_hour_counts = compute_weekday_hour_counts(rows)

        plot_weekday_channel(
            channel_id, weekday_profile, weekday_hour_counts, stats=stats, save=True, output_dir=output_dir, channels_data=CHANNELS_DATA, build_overview_text_func=build_overview_text
        )

        all_channels_plot_data.append(
            {
                "channel_id": channel_id,
                "weekday_profile": weekday_profile,
                "weekday_heatmap": weekday_heatmap,
            }
        )

        all_channels_ranking_data.append(
            {
                "channel_id": channel_id,
                "stats": stats,
            }
        )

    plot_weekday_overview(all_channels_plot_data, save=True, output_dir=output_dir, channels_data=CHANNELS_DATA)
    plot_channel_rankings(all_channels_ranking_data, save=True, output_dir=output_dir, channels_data=CHANNELS_DATA)


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
        process_all_channels(args.start_date, args.end_date)
    else:
        rows = load_ads_data(args.channel_id, args.start_date, args.end_date)
        stats = compute_stats(rows)
        print_stats(args.channel_id, stats)

        if not args.no_plot:
            hourly_profile = compute_hourly_profile(rows)
            plot_hourly_profile(args.channel_id, hourly_profile, stats=stats, output_dir=Path("visualizer_output"), channels_data=CHANNELS_DATA, build_overview_text_func=build_overview_text)
            heatmap = compute_heatmap(rows)
            plot_heatmap(args.channel_id, heatmap, stats=stats, output_dir=Path("visualizer_output"), channels_data=CHANNELS_DATA, build_overview_text_func=build_overview_text)


if __name__ == "__main__":
    main()