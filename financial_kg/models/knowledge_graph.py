"""
Knowledge Graph model for FinancialKG
"""
from typing import List, Dict, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field

from .entity import Entity
from .relationship import Relationship


class KnowledgeGraph(BaseModel):
    """
    Financial Knowledge Graph containing entities and relationships
    
    Similar to ATOM's KnowledgeGraph but with:
    - Sentiment-aware relationships
    - 4-dimensional temporal tracking
    - Financial-specific entity types
    - Multi-source tracking
    """
    
    entities: List[Entity] = Field(
        default_factory=list,
        description="List of entities in the graph"
    )
    
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="List of relationships in the graph"
    )
    
    # Metadata
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When the KG was created"
    )
    
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="When the KG was last updated"
    )
    
    sources: Set[str] = Field(
        default_factory=set,
        description="Set of all source document IDs"
    )
    
    version: str = Field(
        default="1.0.0",
        description="KG version for tracking"
    )
    
    metadata: Dict[str, any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph"""
        # Check if entity already exists
        existing = self.get_entity_by_id(entity.id)
        if existing:
            # Update existing entity
            idx = self.entities.index(existing)
            self.entities[idx] = entity
        else:
            self.entities.append(entity)
        
        # Update sources
        self.sources.update(entity.sources)
        self.updated_at = datetime.now()
    
    def add_relationship(self, relationship: Relationship) -> None:
        """Add a relationship to the graph"""
        # Check if relationship already exists
        existing = self.get_relationship_by_id(relationship.id)
        if existing:
            # Update existing relationship
            idx = self.relationships.index(existing)
            self.relationships[idx] = relationship
        else:
            self.relationships.append(relationship)
        
        # Update sources
        self.sources.update(relationship.properties.sources)
        self.updated_at = datetime.now()
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        for entity in self.entities:
            if entity.id == entity_id:
                return entity
        return None
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get entity by name"""
        for entity in self.entities:
            if entity.name.lower() == name.lower():
                return entity
        return None
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        return [e for e in self.entities if e.type == entity_type]
    
    def get_relationship_by_id(self, rel_id: str) -> Optional[Relationship]:
        """Get relationship by ID"""
        for rel in self.relationships:
            if rel.id == rel_id:
                return rel
        return None
    
    def get_relationships_by_type(self, rel_type: str) -> List[Relationship]:
        """Get all relationships of a specific type"""
        return [r for r in self.relationships if r.predicate == rel_type]
    
    def get_relationships_for_entity(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity (as subject or object)"""
        return [
            r for r in self.relationships
            if r.subject.id == entity_id or r.object.id == entity_id
        ]
    
    def get_outgoing_relationships(self, entity_id: str) -> List[Relationship]:
        """Get outgoing relationships from an entity"""
        return [r for r in self.relationships if r.subject.id == entity_id]
    
    def get_incoming_relationships(self, entity_id: str) -> List[Relationship]:
        """Get incoming relationships to an entity"""
        return [r for r in self.relationships if r.object.id == entity_id]
    
    def merge(self, other: 'KnowledgeGraph') -> 'KnowledgeGraph':
        """
        Merge another KG into this one
        Returns a new merged KnowledgeGraph
        """
        merged = KnowledgeGraph(
            entities=self.entities.copy(),
            relationships=self.relationships.copy(),
            sources=self.sources.copy(),
            metadata=self.metadata.copy()
        )
        
        # Add entities from other
        for entity in other.entities:
            merged.add_entity(entity)
        
        # Add relationships from other
        for relationship in other.relationships:
            merged.add_relationship(relationship)
        
        return merged
    
    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the KG"""
        entity_types = {}
        for entity in self.entities:
            entity_types[entity.type] = entity_types.get(entity.type, 0) + 1
        
        relationship_types = {}
        for rel in self.relationships:
            relationship_types[rel.predicate] = relationship_types.get(rel.predicate, 0) + 1
        
        # Calculate average sentiment
        sentiments = [
            r.properties.sentiment
            for r in self.relationships
            if r.properties.sentiment is not None
        ]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else None
        
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "total_sources": len(self.sources),
            "average_sentiment": avg_sentiment,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def to_dict(self) -> Dict:
        """Convert KG to dictionary format"""
        return {
            "entities": [e.dict() for e in self.entities],
            "relationships": [r.dict() for r in self.relationships],
            "sources": list(self.sources),
            "metadata": self.metadata,
            "stats": self.get_stats()
        }
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: lambda v: list(v)
        }


if __name__ == "__main__":
    from .entity import create_entity
    from .relationship import Relationship, RelationshipProperties
    from .temporal_models import TemporalAttributes
    
    # Create a simple test KG
    kg = KnowledgeGraph()
    
    # Add entities
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={"ticker": "RELIANCE", "sector": "Energy"}
    )
    
    tcs = create_entity(
        entity_type="Company",
        id="TCS",
        name="Tata Consultancy Services",
        properties={"ticker": "TCS", "sector": "IT"}
    )
    
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy",
        properties={"index": "NIFTY_ENERGY"}
    )
    
    kg.add_entity(reliance)
    kg.add_entity(tcs)
    kg.add_entity(energy)
    
    # Add relationship
    rel = Relationship(
        id="REL_001",
        name="OPERATES_IN",
        subject=reliance,
        predicate="OPERATES_IN",
        object=energy,
        properties=RelationshipProperties(
            sentiment=0.7,
            confidence=0.95
        )
    )
    
    kg.add_relationship(rel)
    
    # Print stats
    stats = kg.get_stats()
    print("Knowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
