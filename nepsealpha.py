"""
NepseAlpha Web Scraper
Tertiary data source: https://nepsealpha.com/trading/1/history
"""

import requests
import pandas as pd
import logging
from typing import Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

URL = "https://nepsealpha.com/nepse-data"
ALT_URL = "https://nepsealpha.com/trading/1/history"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}
TIMEOUT = 20


def fetch_from_nepsealpha() -> Optional[pd.DataFrame]:
    """
    Scrape market data from NepseAlpha.
    Returns normalized-ish DataFrame or None on failure.
    """
    for url in (URL, ALT_URL):
        try:
            response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            table = None
            for candidate in soup.find_all("table"):
                header_text = candidate.get_text(separator="|").lower()
                if any(k in header_text for k in ("symbol", "ltp", "close", "change", "volume")):
                    table = candidate
                    break

            if table is None:
                # Try JSON data embedded in script tags
                for script in soup.find_all("script"):
                    if script.string and "symbol" in script.string.lower():
                        import json, re
                        match = re.search(r'\[.*?\]', script.string, re.DOTALL)
                        if match:
                            try:
                                records = json.loads(match.group())
                                if isinstance(records, list) and records:
                                    df = pd.DataFrame(records)
                                    logger.info(
                                        f"NepseAlpha: extracted {len(df)} rows from script tag."
                                    )
                                    return df
                            except Exception:
                                pass
                logger.warning(f"NepseAlpha: No table found at {url}.")
                continue

            rows = []
            headers = []
            for i, row in enumerate(table.find_all("tr")):
                cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                if i == 0:
                    headers = cells
                elif cells:
                    rows.append(cells)

            if not headers or not rows:
                logger.warning(f"NepseAlpha: Empty table at {url}.")
                continue

            aligned = []
            for row in rows:
                if len(row) >= len(headers):
                    aligned.append(row[: len(headers)])
                else:
                    row += [""] * (len(headers) - len(row))
                    aligned.append(row)

            df = pd.DataFrame(aligned, columns=headers)
            logger.info(f"NepseAlpha: scraped {len(df)} rows from {url}.")
            return df

        except requests.exceptions.ConnectionError:
            logger.error(f"NepseAlpha: Connection failed at {url}.")
        except requests.exceptions.Timeout:
            logger.error(f"NepseAlpha: Timeout at {url}.")
        except requests.exceptions.HTTPError as e:
            logger.error(f"NepseAlpha: HTTP error {e} at {url}.")
        except Exception as e:
            logger.error(f"NepseAlpha: Unexpected error at {url} — {e}")

    return None
