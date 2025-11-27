## OqeeAdWatch

OqeeAdWatch is a small watcher that periodically polls the Oqee anti-adskipping API and records every detected ad break in a local SQLite database.

### Getting started

```bash
cd content/posts/ads
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