"""
FinancialKG - Multi-Modal Financial Knowledge Graph Construction

A novel implementation extending itext2kg/ATOM methodology for financial domain,
using Google Gemini models and multi-modal data sources.
"""

__version__ = "0.1.0"
__author__ = "FinancialKG Team"

from .models.entity import Entity, CompanyEntity, PersonEntity, SectorEntity, EventEntity, PolicyEntity, IndicatorEntity, create_entity
from .models.relationship import Relationship, RelationshipProperties, RELATIONSHIP_TYPES
from .models.temporal_models import TemporalAttributes, TimeRange, MarketSession, get_market_session, get_trading_day
from .models.knowledge_graph import KnowledgeGraph

from .utils.config import get_config, Config
from .utils.logging_config import setup_logging, get_logger
from .utils.gemini_client import get_gemini_client, GeminiClient
from .storage import Neo4jStorage

__all__ = [
    # Version
    "__version__",
    "__author__",
    
    # Models
    "Entity",
    "CompanyEntity",
    "PersonEntity",
    "SectorEntity",
    "EventEntity",
    "PolicyEntity",
    "IndicatorEntity",
    "create_entity",
    "Relationship",
    "RelationshipProperties",
    "RELATIONSHIP_TYPES",
    "TemporalAttributes",
    "TimeRange",
    "MarketSession",
    "get_market_session",
    "get_trading_day",
    "KnowledgeGraph",
    
    # Utils
    "get_config",
    "Config",
    "setup_logging",
    "get_logger",
    "get_gemini_client",
    "GeminiClient",
    "Neo4jStorage",
]
