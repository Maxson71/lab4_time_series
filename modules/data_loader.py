"""
Модуль завантаження та підготовки даних Time Series
"""
import requests
import pandas as pd
from datetime import date, timedelta
import os
from pathlib import Path
from typing import Tuple, Optional


class TimeSeriesLoader:
    """Клас для завантаження та підготовки даних часових рядів"""

    def __init__(self, valcode: str = "EUR", days: int = 3 * 365):
        self.valcode = valcode
        self.days = days
        self.base_dir = Path(valcode)

    def fetch_nbu_data(self) -> pd.DataFrame:
        """Завантаження даних з API НБУ"""
        end = date.today()
        start = end - timedelta(days=self.days)

        url = "https://bank.gov.ua/NBU_Exchange/exchange_site"
        params = {
            "start": start.strftime("%Y%m%d"),
            "end": end.strftime("%Y%m%d"),
            "valcode": self.valcode.lower(),
            "sort": "exchangedate",
            "order": "asc",
            "json": "",
        }

        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["exchangedate"], format="%d.%m.%Y")
        df = df.sort_values("date").reset_index(drop=True)
        df = df[["date", "cc", "rate"]]

        return df

    def save_data(self, df: pd.DataFrame) -> None:
        """Збереження даних у форматі CSV та Parquet"""
        data_dir = self.base_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        csv_path = data_dir / f"NBU_{self.valcode}_UAH.csv"
        df.to_csv(csv_path, index=False)

        try:
            parquet_path = data_dir / f"NBU_{self.valcode}_UAH.parquet"
            df.to_parquet(parquet_path, index=False)
        except Exception as e:
            print(f"Warning: Could not save Parquet file: {e}")

    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Завантаження даних з файлу"""
        if file_path is None:
            file_path = self.base_dir / f"data/NBU_{self.valcode}_UAH.csv"

        df = pd.read_csv(file_path, parse_dates=["date"])
        return df

    def get_or_fetch_data(self) -> pd.DataFrame:
        """Завантажити з файлу або отримати з API"""
        csv_path = self.base_dir / f"data/NBU_{self.valcode}_UAH.csv"

        if csv_path.exists():
            print(f"Loading existing data from {csv_path}")
            return self.load_data()
        else:
            print("Fetching fresh data from NBU API...")
            df = self.fetch_nbu_data()
            self.save_data(df)
            return df
