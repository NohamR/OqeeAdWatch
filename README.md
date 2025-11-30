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

The helper `visualizer.py` script prints a quick summary for a single channel:

```bash
uv run python utils/visualizer.py <channel-id>
```

You will see totals, min/max dates, longest breaks, and a per-day breakdown for that channel based on the ads already stored in `ads.sqlite3`. Matplotlib windows display:

- A 24h profile (bars = average ad minutes per day, line = average break count).
- A minute-vs-hour heatmap (white to red) showing how much of each minute is covered by ads on average.

Add `--no-plot` if you only want the textual summary.

### Webhook heartbeat

OqeeAdWatch can send a heartbeat notification every 24 hours to confirm the scraper is still running. To enable it:

1. Copy `.env.example` to `.env`
2. Set `WEBHOOK_URL` to your webhook endpoint (Discord, Slack, etc.)