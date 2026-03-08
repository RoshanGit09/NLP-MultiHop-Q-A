"""
Production Knowledge Graph Builder
Processes all data sources and builds a comprehensive Financial KG
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import asyncio

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from financial_kg.extractors.gemini_atomic_extractor import GeminiAtomicExtractor
from financial_kg.models.knowledge_graph import KnowledgeGraph
from financial_kg.storage.neo4j_storage import Neo4jStorage
from financial_kg.utils.logging_config import get_logger
from financial_kg.utils.config import get_config

# Setup logging
logger = get_logger(__name__)

class ProductionKGBuilder:
    """Builds Knowledge Graph from all available data sources"""
    
    def __init__(self, config_path: str = None):
        """Initialize the production KG builder"""
        self.config = get_config()
        
        # Initialize extractor with Gemini
        self.extractor = GeminiAtomicExtractor()
        
        # Initialize Knowledge Graph
        self.kg = KnowledgeGraph()
        
        # Data paths
        self.data_dir = Path(__file__).parent / 'data'
        
        logger.info("[OK] Production KG Builder initialized successfully")
    
    async def process_news_data(self, max_articles: int = None) -> None:
        """
        Process news articles from Zenodo IN-FINews dataset
        
        Args:
            max_articles: Maximum number of articles to process (None = all)
        """
        logger.info("=" * 80)
        logger.info("PROCESSING NEWS DATA FROM ZENODO")
        logger.info("=" * 80)
        
        news_file = self.data_dir / 'zenodo' / 'IN-FINews Dataset.csv'
        
        if not news_file.exists():
            logger.warning(f"News file not found: {news_file}")
            return
        
        try:
            # Load news data
            df = pd.read_csv(news_file)
            logger.info(f"Loaded {len(df)} news articles")
            
            # Limit if specified
            if max_articles:
                df = df.head(max_articles)
                logger.info(f"Processing first {max_articles} articles")
            
            total = len(df)
            # Process each article using async properly
            for idx, row in df.iterrows():
                try:
                    # Combine title and content
                    text = f"{row['Title']}\n\n{row['Content']}"
                    
                    context = {
                        'source': 'IN-FINews',
                        'date': str(row.get('Date', '')),
                        'url': str(row.get('URL', '')),
                        'article_id': str(idx)
                    }
                    
                    # Extract entities and relationships (properly awaited)
                    entities, relationships = await self.extractor.extract_entities_and_relationships(text, context)
                    
                    self.kg.add_entities(entities)
                    self.kg.add_relationships(relationships)
                    
                    if (idx + 1) % 10 == 0:
                        stats = self.kg.get_stats()
                        logger.info(f"Progress: {idx + 1}/{total} articles | "
                                  f"Entities: {stats['total_entities']} | "
                                  f"Relationships: {stats['total_relationships']}")
                
                except Exception as e:
                    logger.error(f"Error processing article {idx}: {str(e)}")
                    continue
            
            logger.info("[OK] News data processing completed")
            
        except Exception as e:
            logger.error(f"Failed to process news data: {str(e)}")
    
    async def process_stock_data(self, max_stocks: int = 10) -> None:
        """
        Process stock market data from NIFTY-50 dataset
        
        Args:
            max_stocks: Maximum number of stocks to process
        """
        logger.info("=" * 80)
        logger.info("PROCESSING STOCK MARKET DATA")
        logger.info("=" * 80)
        
        stock_dir = self.data_dir / 'kaggle' / 'NIFTY-50 Stock Market Data (2000 - 2021)'
        
        if not stock_dir.exists():
            logger.warning(f"Stock data directory not found: {stock_dir}")
            return
        
        try:
            # Get all CSV files (each represents a company)
            csv_files = list(stock_dir.glob('*.csv'))
            csv_files = [f for f in csv_files if f.name != 'stock_metadata.csv' and f.name != 'NIFTY50_all.csv']
            
            logger.info(f"Found {len(csv_files)} stock files")
            
            # Limit number of stocks
            csv_files = csv_files[:max_stocks]
            
            for stock_file in csv_files:
                try:
                    symbol = stock_file.stem
                    logger.info(f"  Processing stock: {symbol}")
                    
                    df = pd.read_csv(stock_file)
                    recent_data = df.tail(30)
                    
                    summary = self._create_stock_summary(symbol, recent_data)
                    
                    # Extract entities and relationships (properly awaited)
                    entities, relationships = await self.extractor.extract_entities_and_relationships(summary)
                    
                    self.kg.add_entities(entities)
                    self.kg.add_relationships(relationships)
                    
                except Exception as e:
                    logger.error(f"Error processing stock {stock_file.name}: {str(e)}")
                    continue
            
            logger.info("[OK] Stock data processing completed")
            
        except Exception as e:
            logger.error(f"Failed to process stock data: {str(e)}")
    
    def _create_stock_summary(self, symbol: str, df: pd.DataFrame) -> str:
        """Create a natural language summary of stock performance"""
        
        if len(df) == 0:
            return f"{symbol} has no recent trading data."
        
        latest = df.iloc[-1]
        oldest = df.iloc[0]
        
        avg_volume = df['Volume'].mean() if 'Volume' in df.columns else 0
        price_change = ((latest['Close'] - oldest['Close']) / oldest['Close'] * 100) if 'Close' in df.columns else 0
        high_price = df['High'].max() if 'High' in df.columns else 0
        low_price = df['Low'].min() if 'Low' in df.columns else 0
        
        summary = (
            f"{symbol} Stock Performance Summary: "
            f"The company {symbol} traded at {latest['Close']:.2f} on {latest['Date']}. "
            f"Over the recent period, the stock showed a {abs(price_change):.2f}% "
            f"{'increase' if price_change > 0 else 'decrease'}. "
            f"The stock reached a high of {high_price:.2f} and a low of {low_price:.2f}. "
            f"Average daily trading volume was {avg_volume:.0f} shares."
        )
        
        return summary
    
    def save_kg(self, output_dir: str = "output") -> None:
        """Save the Knowledge Graph to JSON"""
        logger.info("=" * 80)
        logger.info("SAVING KNOWLEDGE GRAPH")
        logger.info("=" * 80)
        
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"financial_kg_{timestamp}.json"
        
        self.kg.save_json(str(json_file))
        
        logger.info(f"[OK] Knowledge Graph saved to: {json_file}")
        
        stats = self.kg.get_stats()
        logger.info("\n" + "=" * 80)
        logger.info("KNOWLEDGE GRAPH STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Entities: {stats['total_entities']}")
        logger.info(f"Entity Types: {stats['entity_types']}")
        logger.info(f"Total Relationships: {stats['total_relationships']}")
        logger.info(f"Relationship Types: {stats['relationship_types']}")
        if stats.get('average_sentiment') is not None:
            logger.info(f"Average Sentiment: {stats['average_sentiment']:.2f}")
        logger.info("=" * 80)
    
    def upload_to_neo4j(self) -> bool:
        """
        Upload Knowledge Graph to Neo4j
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("=" * 80)
        logger.info("UPLOADING TO NEO4J")
        logger.info("=" * 80)
        
        try:
            with Neo4jStorage(
                uri=self.config.neo4j.uri,
                username=self.config.neo4j.username,
                password=self.config.neo4j.password
            ) as neo4j_storage:
                neo4j_storage.upload_kg(self.kg)
            
            logger.info("[OK] Successfully uploaded to Neo4j")
            return True
            
        except Exception as e:
            logger.error(f"Failed to upload to Neo4j: {str(e)}")
            logger.info("KG is saved locally in JSON format")
            return False
    
    async def build(self, max_news: int = 50, max_stocks: int = 10, upload_neo4j: bool = True) -> None:
        """
        Complete pipeline: Process all data and build KG
        
        Args:
            max_news: Maximum number of news articles to process
            max_stocks: Maximum number of stocks to process
            upload_neo4j: Whether to upload to Neo4j
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING KNOWLEDGE GRAPH CONSTRUCTION")
        logger.info("=" * 80)
        logger.info(f"Max News Articles: {max_news}")
        logger.info(f"Max Stocks: {max_stocks}")
        logger.info(f"Upload to Neo4j: {upload_neo4j}")
        logger.info("=" * 80 + "\n")
        
        # Process news data
        await self.process_news_data(max_articles=max_news)
        
        # Process stock data
        await self.process_stock_data(max_stocks=max_stocks)
        
        # Save to JSON
        self.save_kg()
        
        # Upload to Neo4j if requested
        if upload_neo4j:
            self.upload_to_neo4j()
        
        logger.info("\n" + "=" * 80)
        logger.info("[OK] KNOWLEDGE GRAPH CONSTRUCTION COMPLETED")
        logger.info("=" * 80)


async def main():
    """Main execution function"""
    
    builder = ProductionKGBuilder()
    
    # Adjust max_news and max_stocks as needed:
    # - max_news: up to 3350 articles available
    # - max_stocks: up to 51 stock files available
    await builder.build(
        max_news=100,
        max_stocks=10,
        upload_neo4j=True
    )


if __name__ == "__main__":
    asyncio.run(main())
