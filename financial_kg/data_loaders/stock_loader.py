"""
Stock data loader for NIFTY-50 and other stock price datasets
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime, date
import glob

from utils.logging_config import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class StockDataLoader:
    """
    Loader for stock market data (NIFTY-50, historical prices, etc.)
    
    Handles OHLCV (Open, High, Low, Close, Volume) data and metadata
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize stock data loader
        
        Args:
            data_dir: Base directory for stock data (defaults to config)
        """
        self.data_dir = data_dir or config.data.data_dir / "nifty50"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"StockDataLoader initialized with data_dir: {self.data_dir}")
    
    def load_stock_csv(self, ticker: str, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load stock data from CSV file
        
        Args:
            ticker: Stock ticker symbol (e.g., 'RELIANCE', 'TCS')
            file_path: Optional custom file path (defaults to data_dir/TICKER.csv)
            
        Returns:
            DataFrame with OHLCV data
        """
        if file_path is None:
            file_path = self.data_dir / f"{ticker}.csv"
        
        if not file_path.exists():
            logger.error(f"Stock data file not found: {file_path}")
            raise FileNotFoundError(f"Stock data for {ticker} not found at {file_path}")
        
        logger.info(f"Loading stock data for {ticker} from {file_path}")
        
        # Load CSV
        df = pd.read_csv(file_path)
        
        # Standardize column names (handle different formats)
        df = self._standardize_columns(df)
        
        # Parse dates
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
        
        logger.info(f"Loaded {len(df)} rows for {ticker}")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different data formats"""
        # Common column mappings
        column_mapping = {
            'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume',
            'adj close': 'Adj Close',
            'prev close': 'Prev Close',
        }
        
        # Rename columns (case-insensitive)
        df.columns = df.columns.str.strip()
        for old, new in column_mapping.items():
            matching = [col for col in df.columns if col.lower() == old]
            if matching:
                df = df.rename(columns={matching[0]: new})
        
        return df
    
    def load_all_nifty50(self) -> Dict[str, pd.DataFrame]:
        """
        Load all NIFTY-50 stocks
        
        Returns:
            Dictionary mapping ticker → DataFrame
        """
        logger.info("Loading all NIFTY-50 stocks...")
        
        # Find all CSV files
        csv_files = glob.glob(str(self.data_dir / "*.csv"))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_dir}")
            return {}
        
        stocks = {}
        for file_path in csv_files:
            ticker = Path(file_path).stem
            try:
                stocks[ticker] = self.load_stock_csv(ticker, Path(file_path))
            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")
        
        logger.info(f"Loaded {len(stocks)} stocks")
        return stocks
    
    def get_price_on_date(
        self, 
        ticker: str, 
        target_date: Union[datetime, date, str],
        df: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get price data for a specific date
        
        Args:
            ticker: Stock ticker
            target_date: Target date
            df: Optional pre-loaded DataFrame
            
        Returns:
            Dictionary with OHLCV data or None if not found
        """
        if df is None:
            df = self.load_stock_csv(ticker)
        
        # Parse date
        if isinstance(target_date, str):
            target_date = pd.to_datetime(target_date)
        
        # Find matching row
        mask = df['Date'] == target_date
        rows = df[mask]
        
        if len(rows) == 0:
            logger.warning(f"No data for {ticker} on {target_date}")
            return None
        
        row = rows.iloc[0]
        return {
            'open': float(row.get('Open', 0)),
            'high': float(row.get('High', 0)),
            'low': float(row.get('Low', 0)),
            'close': float(row.get('Close', 0)),
            'volume': float(row.get('Volume', 0)),
            'date': target_date
        }
    
    def calculate_price_change(
        self,
        ticker: str,
        start_date: Union[datetime, date, str],
        end_date: Union[datetime, date, str]
    ) -> Optional[float]:
        """
        Calculate percentage price change between two dates
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Percentage change (e.g., 5.5 for +5.5%)
        """
        start_data = self.get_price_on_date(ticker, start_date)
        end_data = self.get_price_on_date(ticker, end_date)
        
        if start_data is None or end_data is None:
            return None
        
        change = ((end_data['close'] - start_data['close']) / start_data['close']) * 100
        return change


# NIFTY-50 company list
NIFTY50_COMPANIES = {
    'RELIANCE': 'Reliance Industries Ltd.',
    'TCS': 'Tata Consultancy Services Ltd.',
    'HDFCBANK': 'HDFC Bank Ltd.',
    'INFY': 'Infosys Ltd.',
    'ICICIBANK': 'ICICI Bank Ltd.',
}
