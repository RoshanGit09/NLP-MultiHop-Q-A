"""
News data loader for financial news datasets (HuggingFace, Zenodo, etc.)
"""
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from datetime import datetime
import json

from datasets import load_dataset, Dataset
from utils.logging_config import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class NewsDataLoader:
    """
    Loader for financial news datasets
    
    Supports:
    - HuggingFace datasets (Indian Financial News, Headlines, NIFTY)
    - JSON/CSV files (IN-FINews from Zenodo)
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize news data loader
        
        Args:
            data_dir: Base directory for news data
        """
        self.data_dir = data_dir or config.data.data_dir / "news"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"NewsDataLoader initialized with data_dir: {self.data_dir}")
    
    def load_huggingface_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        cache_dir: Optional[Path] = None
    ) -> Dataset:
        """
        Load dataset from HuggingFace
        
        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split (train, test, validation)
            cache_dir: Cache directory for downloaded data
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading HuggingFace dataset: {dataset_name}")
        
        cache = cache_dir or config.data.cache_dir / "huggingface"
        
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=str(cache)
            )
            logger.info(f"Loaded {len(dataset)} items from {dataset_name}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            raise
    
    def load_indian_financial_news(
        self,
        max_items: Optional[int] = None
    ) -> List[Dict]:
        """
        Load Indian Financial News dataset (26k articles)
        
        Args:
            max_items: Maximum number of items to load
            
        Returns:
            List of news dictionaries
        """
        dataset = self.load_huggingface_dataset("kdave/Indian_Financial_News")
        
        news_list = []
        for idx, item in enumerate(dataset):
            if max_items and idx >= max_items:
                break
            
            news_list.append({
                'title': item.get('title', ''),
                'content': item.get('content', ''),
                'date': item.get('date', ''),
                'source': 'Indian_Financial_News',
                'id': f"IFN_{idx}"
            })
        
        logger.info(f"Loaded {len(news_list)} articles from Indian Financial News")
        return news_list
    
    def load_financial_headlines(
        self,
        max_items: Optional[int] = None
    ) -> List[Dict]:
        """
        Load Financial News Headlines dataset
        
        Args:
            max_items: Maximum number of items to load
            
        Returns:
            List of headline dictionaries
        """
        dataset = self.load_huggingface_dataset("steve1989/financial_news_headlines")
        
        headlines = []
        for idx, item in enumerate(dataset):
            if max_items and idx >= max_items:
                break
            
            headlines.append({
                'headline': item.get('headline', ''),
                'date': item.get('date', ''),
                'sentiment': item.get('sentiment', None),
                'source': 'Financial_Headlines',
                'id': f"FH_{idx}"
            })
        
        logger.info(f"Loaded {len(headlines)} headlines")
        return headlines
    
    def load_nifty_headlines(
        self,
        max_items: Optional[int] = None
    ) -> List[Dict]:
        """
        Load NIFTY News Headlines dataset
        
        Args:
            max_items: Maximum number of items to load
            
        Returns:
            List of NIFTY headline dictionaries
        """
        dataset = self.load_huggingface_dataset("raeidsaqur/NIFTY")
        
        headlines = []
        for idx, item in enumerate(dataset):
            if max_items and idx >= max_items:
                break
            
            headlines.append({
                'headline': item.get('headline', ''),
                'date': item.get('date', ''),
                'source': 'NIFTY_Headlines',
                'id': f"NH_{idx}"
            })
        
        logger.info(f"Loaded {len(headlines)} NIFTY headlines")
        return headlines
    
    def load_json_news(self, file_path: Path) -> List[Dict]:
        """
        Load news from JSON file (e.g., IN-FINews)
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of news dictionaries
        """
        logger.info(f"Loading news from JSON: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            news = json.load(f)
        
        if isinstance(news, dict):
            news = [news]
        
        logger.info(f"Loaded {len(news)} articles from JSON")
        return news
    
    def filter_by_date_range(
        self,
        news_list: List[Dict],
        start_date: str,
        end_date: str,
        date_field: str = 'date'
    ) -> List[Dict]:
        """
        Filter news by date range
        
        Args:
            news_list: List of news dictionaries
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            date_field: Field name containing date
            
        Returns:
            Filtered news list
        """
        from dateutil.parser import parse
        
        start = parse(start_date)
        end = parse(end_date)
        
        filtered = []
        for item in news_list:
            if date_field in item and item[date_field]:
                try:
                    item_date = parse(item[date_field])
                    if start <= item_date <= end:
                        filtered.append(item)
                except:
                    pass
        
        logger.info(f"Filtered {len(filtered)} items in date range {start_date} to {end_date}")
        return filtered
    
    def search_by_keyword(
        self,
        news_list: List[Dict],
        keywords: List[str],
        fields: List[str] = ['title', 'content', 'headline']
    ) -> List[Dict]:
        """
        Search news by keywords
        
        Args:
            news_list: List of news dictionaries
            keywords: List of keywords to search
            fields: Fields to search in
            
        Returns:
            Filtered news list
        """
        keywords_lower = [k.lower() for k in keywords]
        
        filtered = []
        for item in news_list:
            for field in fields:
                if field in item and item[field]:
                    text = str(item[field]).lower()
                    if any(keyword in text for keyword in keywords_lower):
                        filtered.append(item)
                        break
        
        logger.info(f"Found {len(filtered)} items matching keywords: {keywords}")
        return filtered


if __name__ == "__main__":
    # Test the loader
    loader = NewsDataLoader()
    
    # Try loading datasets
    try:
        # Load a small sample
        news = loader.load_indian_financial_news(max_items=10)
        print(f"Loaded {len(news)} news articles")
        if news:
            print(f"Sample: {news[0]['title'][:100]}...")
    except Exception as e:
        print(f"Error loading news: {e}")
        print("Make sure you have internet connection and HF token configured")
