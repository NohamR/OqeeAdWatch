"""Database and API scraping utilities for OqeeAdWatch."""

from datetime import datetime
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests


SERVICE_PLAN_API_URL = "https://api.oqee.net/api/v6/service_plan"
DB_PATH = Path(__file__).resolve().parent.parent / "ads.sqlite3"
REQUEST_TIMEOUT = 10
POLL_INTERVAL_SECONDS = 30 * 60  # 30 minutes


logger = logging.getLogger(__name__)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a SQLite connection configured for our ad tracking."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the ads table if it does not already exist."""

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ads (
            channel_id TEXT NOT NULL,
            start_ts INTEGER NOT NULL,
            end_ts INTEGER NOT NULL,
            ad_date TEXT NOT NULL,
            PRIMARY KEY (channel_id, start_ts, end_ts)
        )
        """
    )


def record_ad_break(
    conn: sqlite3.Connection,
    channel_id: str,
    start_ts: int,
    end_ts: int,
) -> bool:
    """Insert an ad break if it is not already stored."""

    ad_date = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO ads (channel_id, start_ts, end_ts, ad_date)
                VALUES (?, ?, ?, ?)
                """,
                (channel_id, start_ts, end_ts, ad_date),
            )
            logger.debug(
                "Ad break recorded in database",
                extra={
                    "channel_id": channel_id,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                },
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_ads_for_channel(
    conn: sqlite3.Connection, channel_id: str, limit: Optional[int] = None
) -> List[sqlite3.Row]:
    """Return the most recent ad breaks for a channel."""

    query = (
        "SELECT channel_id, start_ts, end_ts, ad_date "
        "FROM ads WHERE channel_id = ? ORDER BY start_ts DESC"
    )
    if limit:
        query += " LIMIT ?"
        params = (channel_id, limit)
    else:
        params = (channel_id,)
    return conn.execute(query, params).fetchall()


def fetch_service_plan():
    """Fetch the channel list supporting anti-ad skipping."""

    api_url = SERVICE_PLAN_API_URL
    try:
        logger.info("Loading channel list from the Oqee API...")
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if not data.get("success") or "channels" not in data.get("result", {}):
            logger.error("Error: Unexpected API response format.")
            return None

        channels_data = data["result"]["channels"]
        return channels_data

    except requests.exceptions.RequestException as exc:
        logger.error("A network error occurred: %s", exc)
        return None
    except ValueError:
        logger.error("Error while parsing the JSON response.")
        return None


def fetch_and_parse_ads(channel_id: str, conn: sqlite3.Connection) -> None:
    """Collect ad breaks for a channel and persist the unseen entries."""

    total_seconds = 0
    url = f"https://api.oqee.net/api/v1/live/anti_adskipping/{channel_id}"

    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    data = response.json()

    periods = data.get('result', {}).get('periods', [])

    if not periods:
        logger.info("No periods data found for channel %s", channel_id)
        return

    logger.debug(
        "%s | %s | %s",
        "Start Time".ljust(22),
        "End Time".ljust(22),
        "Duration",
    )
    logger.debug("-" * 60)

    ad_count = 0
    stored_ads = 0
    for item in periods:
        if item.get('type') == 'ad_break':
            start_ts = item.get('start_time')
            end_ts = item.get('end_time')

            if start_ts is None or end_ts is None:
                logger.warning("Skipping ad break with missing timestamps: %s", item)
                continue

            ad_count += 1
            duration = end_ts - start_ts

            start_date = datetime.fromtimestamp(start_ts).strftime('%Y-%m-%d %H:%M:%S')
            end_date = datetime.fromtimestamp(end_ts).strftime('%Y-%m-%d %H:%M:%S')

            logger.debug(
                "%s | %s | %ss",
                start_date.ljust(22),
                end_date.ljust(22),
                duration,
            )

            total_seconds += duration

            if record_ad_break(conn, channel_id, start_ts, end_ts):
                stored_ads += 1

    logger.debug("-" * 60)
    logger.info("Total ad breaks found: %s", ad_count)
    logger.debug(
        "Total ad duration: %smin %ss",
        total_seconds // 60,
        total_seconds % 60,
    )
    logger.info("New ad entries stored: %s", stored_ads)


def run_collection_cycle(conn: sqlite3.Connection) -> None:
    """Fetch ads for all eligible channels once."""

    channels_data = fetch_service_plan()
    if not channels_data:
        logger.warning("No channel data available for this cycle")
        return

    for channel_id, channel_info in channels_data.items():
        if not channel_info.get("enable_anti_adskipping"):
            continue

        logger.info(
            "Analyzing ads for channel: %s (ID: %s)",
            channel_info.get("name"),
            channel_id,
        )
        try:
            fetch_and_parse_ads(channel_id, conn)
        except requests.RequestException as exc:
            logger.error("Network error for channel %s: %s", channel_id, exc)
        except Exception:  # pylint: disable=broad-exception-caught
            logger.exception("Unexpected error for channel %s", channel_id)
