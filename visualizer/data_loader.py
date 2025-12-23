import sqlite3
from typing import Sequence, List, Optional
from pathlib import Path
import sys

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from utils.scrap import DB_PATH, get_connection

Row = Sequence

def load_ads_data(
    channel_id: str, start_date: Optional[str] = None, end_date: Optional[str] = None
) -> List[Row]:
    """Load ad break data from the database for a given channel and date range."""
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
    except sqlite3.OperationalError as exc:
        raise SystemExit(
            "SQLite query failed. Ensure the collector ran at least once (table 'ads' must exist)."
        ) from exc
    finally:
        conn.close()


def list_channels() -> List[str]:
    """List all channel IDs present in the database."""
    conn = get_connection(DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT DISTINCT channel_id FROM ads ORDER BY channel_id ASC"
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()