"""
Point-in-Time Data Store
========================
Bi-temporal data store ensuring no look-ahead bias.
Records both as-of date and knowledge date for all data.
"""

import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class PointInTimeStore:
    """
    Bi-temporal data store using SQLite.

    Every record has:
      - ticker: security identifier
      - field: data field name
      - value: the data value
      - as_of_date: when the data pertains to
      - knowledge_date: when the data became available
    """

    def __init__(self, db_path: str = "pit_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pit_data (
                ticker TEXT NOT NULL,
                field TEXT NOT NULL,
                value REAL,
                as_of_date TEXT NOT NULL,
                knowledge_date TEXT NOT NULL,
                source TEXT DEFAULT 'unknown',
                PRIMARY KEY (ticker, field, as_of_date)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge
            ON pit_data (knowledge_date, ticker, field)
        """)
        conn.commit()
        conn.close()

    def insert(self, ticker: str, field: str, value: float,
               as_of_date: str, knowledge_date: str,
               source: str = 'unknown'):
        """Insert a single data point."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO pit_data
            (ticker, field, value, as_of_date, knowledge_date, source)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (ticker, field, value, as_of_date, knowledge_date, source))
        conn.commit()
        conn.close()

    def insert_dataframe(self, df: pd.DataFrame):
        """
        Bulk insert from DataFrame.

        Expected columns: ticker, field, value, as_of_date, knowledge_date
        """
        conn = sqlite3.connect(self.db_path)
        records = df[['ticker', 'field', 'value', 'as_of_date',
                       'knowledge_date']].values.tolist()
        conn.executemany("""
            INSERT OR REPLACE INTO pit_data
            (ticker, field, value, as_of_date, knowledge_date)
            VALUES (?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(records)} records into PIT store")

    def query(self, tickers: list, fields: list,
              as_of: str) -> pd.DataFrame:
        """
        Query data available as of a given date.

        This is the critical method: it only returns data where
        knowledge_date <= as_of, ensuring no look-ahead bias.

        Parameters
        ----------
        tickers : list
            List of tickers to query.
        fields : list
            List of fields to query.
        as_of : str
            The backtest date. Only data known on or before this date
            is returned.

        Returns
        -------
        pd.DataFrame
            Pivoted DataFrame with tickers as rows and fields as columns.
            Uses the most recent available data for each ticker/field.
        """
        conn = sqlite3.connect(self.db_path)
        placeholders_t = ','.join(['?' for _ in tickers])
        placeholders_f = ','.join(['?' for _ in fields])

        query = f"""
            SELECT ticker, field, value, as_of_date
            FROM pit_data
            WHERE ticker IN ({placeholders_t})
              AND field IN ({placeholders_f})
              AND knowledge_date <= ?
            ORDER BY as_of_date DESC
        """
        params = tickers + fields + [as_of]
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df.empty:
            return pd.DataFrame(columns=fields, index=tickers)

        # Take the most recent value for each ticker/field
        df = df.drop_duplicates(subset=['ticker', 'field'], keep='first')
        pivot = df.pivot(index='ticker', columns='field', values='value')
        return pivot.reindex(index=tickers, columns=fields)

    def get_record_count(self) -> int:
        """Get total records in store."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM pit_data").fetchone()[0]
        conn.close()
        return count
