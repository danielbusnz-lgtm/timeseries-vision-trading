import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np


class StockDataFetcher:
    def __init__(self, ticker: str, start_date: str = None, end_date: str = None):
        self.ticker = ticker

        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        self.start_date = start_date
        self.end_date = end_date
        self.data = None


    def fetch(self) -> pd.DataFrame:
        print(f"Fetching {self.ticker} from {self.start_date} to {self.end_date}")
        stock = yf.Ticker(self.ticker)
        df = stock.history(start=self.start_date, end=self.end_date)

        df.columns = [col.lower() for col in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.dropna()

        print(f" Downloaded {len(df)} rows")
        self.data = df
        return df


class GramianAngularField:
    def __init__(self, method='summation', image_size=None):
        self.method = method
        self.image_size = image_size

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        """Normalize time series to [-1, 1] range."""
        X_min, X_max = np.min(X), np.max(X)
        if X_max - X_min == 0:
            return np.zeros_like(X)
        return 2 * (X - X_min) / (X_max - X_min) - 1

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform time series to GAF image."""
        X_norm = self._normalize(X)

        phi = np.arccos(X_norm)

        n = len(X_norm)

        if self.method == 'summation':
            gaf = np.cos(phi[:, np.newaxis] + phi[np.newaxis, :])
        else:
            gaf = np.sin(phi[:, np.newaxis] - phi[np.newaxis, :])

        # Resize if needed
        if self.image_size and self.image_size != n:
            from PIL import Image
            # Convert to 0-255 color values
            img = Image.fromarray(((gaf + 1) * 127.5).astype(np.uint8))
            # Resize
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            # Convert back to GAF format
            gaf = np.array(img).astype(np.float32) / 127.5 - 1

        return gaf
