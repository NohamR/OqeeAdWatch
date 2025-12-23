## OqeeAdWatch

OqeeAdWatch is a small watcher that periodically polls the Oqee anti-adskipping API and records every detected ad break in a local SQLite database.

### Getting started

```bash
git clone https://github.com/NohamR/OqeeAdWatch && cd OqeeAdWatch
uv sync
```

Add the debug tooling (Pylint) when needed:

```bash
uv sync --group debug
```

```bash
uv run main.py
```

The script will start a collection cycle immediately, iterate over every eligible channel, then sleep for 30 minutes before repeating. Use `CTRL+C` to exit cleanly.

### Database layout

The database lives next to the script and contains a single `ads` table:

| column      | type    | notes                                |
| ----------- | ------- | ------------------------------------ |
| channel_id  | TEXT    | Oqee channel identifier              |
| start_ts    | INTEGER | UNIX timestamp for the start of ad   |
| end_ts      | INTEGER | UNIX timestamp for the end of ad     |
| ad_date     | TEXT    | Convenience YYYY-MM-DD string        |

The primary key `(channel_id, start_ts, end_ts)` prevents duplicates when the API returns the same break multiple times.

### Visualizing collected ads

The `visualizer/main.py` script analyzes and visualizes ad data from the database:

```bash
# Process all channels (default)
uv run ./visualizer/main.py

# Process a specific channel
uv run ./visualizer/main.py <channel-id>

# Filter by date range
uv run ./visualizer/main.py --start-date 2025-11-28 --end-date 2025-12-21

# Single channel with date filter
uv run ./visualizer/main.py <channel-id> --start-date 2025-11-28
```

**Single channel mode** displays:
- Totals, min/max dates, longest breaks, and a per-day breakdown
- A 24h profile (bars = average ad minutes per day, line = average break count)
- A minute-vs-hour heatmap showing ad coverage

**All channels mode** generates additional visualizations saved to `visualizer_output/`:
- Combined hourly profile and heatmap for each channel
- Weekday analysis per channel (ad breaks by day of week, weekday×hour heatmap)
- Weekly ad patterns overview across all channels
- **Channel rankings** comparing all channels by:
  - Total number of ads
  - Total ad duration
  - Longest single ad break

Add `--no-plot` if you only want the textual summary.

> **Note:** Ad breaks longer than 30 minutes are automatically filtered out as they are likely errors.

### Webhook heartbeat

OqeeAdWatch can send a heartbeat notification every 24 hours to confirm the scraper is still running. To enable it:

1. Copy `.env.example` to `.env`
2. Set `WEBHOOK_URL` to your webhook endpoint (Discord, Slack, etc.)