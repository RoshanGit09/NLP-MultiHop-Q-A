"""
Financial relationship model with sentiment and temporal attributes
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

from .entity import Entity
from .temporal_models import TemporalAttributes, MarketSession


class RelationshipProperties(BaseModel):
    """
    Properties for financial relationships
    Extends basic properties with sentiment and financial-specific attributes
    """
    
    # Sentiment (key innovation over ATOM)
    sentiment: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Sentiment score: -1 (very negative) to +1 (very positive)"
    )
    
    sentiment_label: Optional[str] = Field(
        default=None,
        description="Sentiment label: positive, negative, neutral"
    )
    
    # Impact magnitude
    impact_magnitude: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Magnitude of impact (0 = no impact, 1 = maximum impact)"
    )
    
    # Confidence
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence score"
    )
    
    # Sources
    sources: List[str] = Field(
        default_factory=list,
        description="Source document/URL IDs"
    )
    
    # Financial-specific properties
    price_change_percent: Optional[float] = Field(
        default=None,
        description="Price change % associated with this relationship"
    )
    
    volume_change_percent: Optional[float] = Field(
        default=None,
        description="Trading volume change %"
    )
    
    correlation_coefficient: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="For CORRELATES_WITH relationships"
    )
    
    # Metadata
    prominence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Prominence of mention (e.g., headline vs buried in article)"
    )
    
    # Custom properties
    custom_properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom properties"
    )
    
    class Config:
        arbitrary_types_allowed = True


class Relationship(BaseModel):
    """
    Financial relationship with sentiment and 4-dimensional temporal tracking
    
    Key innovations over ATOM:
    1. Sentiment scoring on relationships
    2. 4-dimensional temporal model (vs ATOM's 2-time)
    3. Financial-specific properties (price impact, correlation, etc.)
    4. Market session awareness
    """
    
    id: str = Field(description="Unique relationship identifier")
    name: str = Field(description="Relationship name/predicate")
    
    # Triple
    subject: Entity = Field(description="Subject entity")
    predicate: str = Field(description="Predicate/relationship type")
    object: Entity = Field(description="Object entity")
    
    # Properties
    properties: RelationshipProperties = Field(
        default_factory=RelationshipProperties,
        description="Relationship properties"
    )
    
    # Temporal attributes (4-dimensional model)
    temporal: TemporalAttributes = Field(
        default_factory=TemporalAttributes,
        description="Temporal attributes with 4-time model"
    )
    
    # Embedding
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding of the relationship"
    )
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def __str__(self):
        sentiment_str = ""
        if self.properties.sentiment is not None:
            sentiment_str = f" [sentiment: {self.properties.sentiment:.2f}]"
        
        return f"({self.subject.name}, {self.predicate}, {self.object.name}){sentiment_str}"
    
    def to_quintuple(self) -> tuple:
        """
        Convert to ATOM-style quintuple format (subject, predicate, object, t_start, t_end)
        Using t_announce as t_start equivalent and t_effective as t_end equivalent
        """
        t_start = self.temporal.t_announce.isoformat() if self.temporal.t_announce else "."
        t_end = self.temporal.t_effective.isoformat() if self.temporal.t_effective else "."
        
        return (
            self.subject.name,
            self.predicate,
            self.object.name,
            t_start,
            t_end
        )
    
    def add_observation(self, obs_date: datetime) -> None:
        """Add an observation timestamp (for incremental updates)"""
        self.temporal.add_observation(obs_date)
        self.updated_at = datetime.now()
    
    def update_sentiment(self, new_sentiment: float, source: Optional[str] = None) -> None:
        """
        Update sentiment score (can be used for aggregating multiple sources)
        
        Args:
            new_sentiment: New sentiment value
            source: Optional source identifier
        """
        if self.properties.sentiment is None:
            self.properties.sentiment = new_sentiment
        else:
            # Average with existing sentiment
            self.properties.sentiment = (self.properties.sentiment + new_sentiment) / 2.0
        
        # Update label
        if self.properties.sentiment > 0.2:
            self.properties.sentiment_label = "positive"
        elif self.properties.sentiment < -0.2:
            self.properties.sentiment_label = "negative"
        else:
            self.properties.sentiment_label = "neutral"
        
        if source:
            self.properties.sources.append(source)
        
        self.updated_at = datetime.now()
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            datetime: lambda v: v.isoformat()
        }


# Common relationship types in financial domain
RELATIONSHIP_TYPES = {
    # Organizational
    "OPERATES_IN": "Company operates in Sector",
    "LISTED_ON": "Company listed on Exchange",
    "LED_BY": "Company led by Person",
    "OWNS": "Entity owns Entity",
    "SUBSIDIARY_OF": "Company is subsidiary of Company",
    
    # Events
    "ANNOUNCED": "Entity announced Event",
    "REPORTED": "Company reported Metric/Result",
    "LAUNCHED": "Company launched Product/Service",
    
    # Impact
    "AFFECTS": "Event/Policy affects Company/Sector",
    "IMPACTS": "Policy impacts Sector/Company",
    "INFLUENCES": "Indicator influences Market",
    
    # Financial
    "CORRELATES_WITH": "Company correlates with Company",
    "COMPETES_WITH": "Company competes with Company",
    "SUPPLIES_TO": "Company supplies to Company",
    "CUSTOMER_OF": "Company is customer of Company",
    
    # News/Media
    "MENTIONED_IN": "Entity mentioned in News",
    "COMMENTED_ON": "Person commented on Event/Company",
    
    # Market
    "PRICE_MOVEMENT": "Company experienced Price Event",
    "VOLUME_SPIKE": "Company experienced Volume Event",
}


if __name__ == "__main__":
    from .entity import create_entity
    from datetime import timedelta
    
    # Create test entities
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={"ticker": "RELIANCE"}
    )
    
    energy_sector = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy",
        properties={"index": "NIFTY_ENERGY"}
    )
    
    # Create relationship
    now = datetime.now()
    rel = Relationship(
        id="REL_001",
        name="OPERATES_IN",
        subject=reliance,
        predicate="OPERATES_IN",
        object=energy_sector,
        properties=RelationshipProperties(
            sentiment=0.7,
            sentiment_label="positive",
            confidence=0.95,
            sources=["annual_report_2024"]
        ),
        temporal=TemporalAttributes(
            t_announce=now - timedelta(days=365),
            t_observe=now
        )
    )
    
    print(f"Relationship: {rel}")
    print(f"Quintuple: {rel.to_quintuple()}")
    print(f"Sentiment: {rel.properties.sentiment_label} ({rel.properties.sentiment})")
