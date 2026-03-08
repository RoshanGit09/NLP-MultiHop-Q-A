"""
Knowledge Graph Builder - Main orchestrator for building financial KGs
"""
import asyncio
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime

from ..utils.logging_config import get_logger
from ..models.knowledge_graph import KnowledgeGraph
from ..models.entity import Entity
from ..models.relationship import Relationship
from ..extractors.gemini_atomic_extractor import GeminiAtomicExtractor
from ..data_loaders.news_loader import NewsDataLoader
from ..data_loaders.stock_loader import StockDataLoader
from ..storage.neo4j_storage import Neo4jStorage

logger = get_logger(__name__)


class FinancialKGBuilder:
    """
    Main orchestrator for building Financial Knowledge Graphs
    
    Coordinates:
    1. Data loading (news + stock data)
    2. Atomic fact extraction (Gemini)
    3. Entity resolution and merging
    4. Temporal relationship construction
    5. KG storage (Neo4j)
    """
    
    def __init__(
        self,
        use_gemini: bool = True,
        use_neo4j: bool = True,
        batch_size: int = 10
    ):
        """
        Initialize KG builder
        
        Args:
            use_gemini: Whether to use Gemini for extraction
            use_neo4j: Whether to store in Neo4j
            batch_size: Batch size for processing
        """
        self.use_gemini = use_gemini
        self.use_neo4j = use_neo4j
        self.batch_size = batch_size
        
        # Initialize components
        self.kg = KnowledgeGraph()
        self.news_loader = NewsDataLoader()
        self.stock_loader = StockDataLoader()
        
        if use_gemini:
            self.extractor = GeminiAtomicExtractor()
        
        if use_neo4j:
            try:
                self.neo4j = Neo4jStorage()
            except Exception as e:
                logger.warning(f"Neo4j not available: {e}")
                self.use_neo4j = False
        
        logger.info("FinancialKGBuilder initialized")
    
    async def build_kg_from_news(
        self,
        news_articles: List[Dict],
        max_articles: Optional[int] = None
    ) -> KnowledgeGraph:
        """
        Build knowledge graph from news articles
        
        Args:
            news_articles: List of news article dictionaries
            max_articles: Maximum articles to process
            
        Returns:
            Built knowledge graph
        """
        logger.info(f"Building KG from {len(news_articles)} news articles")
        
        if max_articles:
            news_articles = news_articles[:max_articles]
        
        # Process articles in batches
        for i in range(0, len(news_articles), self.batch_size):
            batch = news_articles[i:i + self.batch_size]
            await self._process_news_batch(batch)
        
        logger.info(f"KG built with {len(self.kg.entities)} entities and {len(self.kg.relationships)} relationships")
        return self.kg
    
    async def _process_news_batch(self, articles: List[Dict]) -> None:
        """Process a batch of news articles"""
        
        for article in articles:
            try:
                await self._process_single_article(article)
            except Exception as e:
                logger.error(f"Failed to process article: {e}")
                continue
    
    async def _process_single_article(self, article: Dict) -> None:
        """Process a single news article"""
        
        title = article.get('title', '')
        content = article.get('content', '')
        date_str = article.get('date', '')
        source = article.get('source', 'unknown')
        
        # Combine title and content
        text = f"{title}\\n{content}".strip()
        
        if not text:
            return
        
        # Create context
        context = {
            'date': date_str,
            'source': source,
            'article_id': article.get('id', '')
        }
        
        if self.use_gemini:
            # Extract using Gemini - pass raw text directly
            entities, relationships = await self.extractor.extract_entities_and_relationships(text, context)
        else:
            # Use simple rule-based extraction
            entities, relationships = self._simple_extraction(text, context)
        
        # Add to knowledge graph
        for entity in entities:
            self.kg.add_entity(entity)
        
        for relationship in relationships:
            self.kg.add_relationship(relationship)
    
    def _simple_extraction(self, text: str, context: Dict) -> tuple[List[Entity], List[Relationship]]:
        """Simple rule-based extraction (fallback when Gemini not available)"""
        # This is a placeholder - implement basic extraction rules
        entities = []
        relationships = []
        
        # TODO: Implement basic NER and relationship extraction
        # For now, return empty lists
        
        return entities, relationships
    
    async def enhance_with_stock_data(self, stock_data_dir: Path) -> None:
        """
        Enhance KG with stock market data
        
        Args:
            stock_data_dir: Directory containing stock CSV files
        """
        logger.info("Enhancing KG with stock market data")
        
        try:
            # Load all stock data
            self.stock_loader.data_dir = stock_data_dir
            all_stocks = self.stock_loader.load_all_nifty50()
            
            for ticker, df in all_stocks.items():
                self._add_stock_entities_and_relationships(ticker, df)
                
        except Exception as e:
            logger.error(f"Failed to enhance with stock data: {e}")
    
    def _add_stock_entities_and_relationships(self, ticker: str, df) -> None:
        """Add stock-related entities and price relationships"""
        # This would add stock price entities and relationships
        # Implementation depends on your specific requirements
        pass
    
    async def save_to_neo4j(self) -> bool:
        """
        Save knowledge graph to Neo4j
        
        Returns:
            Success status
        """
        if not self.use_neo4j:
            logger.warning("Neo4j not configured")
            return False
        
        try:
            self.neo4j.visualize_graph(self.kg)
            logger.info("Knowledge graph saved to Neo4j")
            return True
        except Exception as e:
            logger.error(f"Failed to save to Neo4j: {e}")
            return False
    
    def save_to_json(self, output_path: Path) -> bool:
        """
        Save knowledge graph to JSON file
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            Success status
        """
        try:
            self.kg.save_json(output_path)
            logger.info(f"Knowledge graph saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save JSON: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get KG building statistics"""
        stats = self.kg.get_stats()
        stats['built_at'] = datetime.now().isoformat()
        return stats
