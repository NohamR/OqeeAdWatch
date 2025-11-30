"""Background ad-break collector that stores anti-adskipping periods in SQLite."""

from datetime import datetime
import logging
import os
import time

import requests
from dotenv import load_dotenv

from utils.scrap import get_connection, init_db, run_collection_cycle, POLL_INTERVAL_SECONDS


load_dotenv()

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_SECONDS = 24 * 60 * 60  # 24 hours


def notify_webhook(message: str) -> bool:
    """Send a notification to the GitHub webhook if configured."""
    webhook_url = os.getenv("WEBHOOK_URL")
    if not webhook_url:
        logger.debug("No WEBHOOK_URL configured, skipping notification")
        return False

    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
        logger.info("Webhook notification sent successfully")
        return True
    except requests.RequestException as exc:
        logger.warning("Failed to send webhook notification: %s", exc)
        return False


def main() -> None:
    """Entrypoint for CLI execution."""

    env_level = os.getenv("LOGLEVEL", "INFO").upper()
    log_level = getattr(logging, env_level, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conn = get_connection()
    init_db(conn)

    last_heartbeat = 0.0

    try:
        while True:
            cycle_started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info("=== New collection started at %s ===", cycle_started)
            try:
                run_collection_cycle(conn)
            except Exception:  # pylint: disable=broad-exception-caught
                logger.exception("Critical error during collection")

            # Send heartbeat notification every 24 hours
            now = time.time()
            if now - last_heartbeat >= HEARTBEAT_INTERVAL_SECONDS:
                notify_webhook(
                    f"🟢 OqeeAdWatch is running - heartbeat at {cycle_started}"
                )
                last_heartbeat = now

            logger.info(
                "Pausing for %s minutes before the next collection...",
                POLL_INTERVAL_SECONDS // 60,
            )
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user. Closing...")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
