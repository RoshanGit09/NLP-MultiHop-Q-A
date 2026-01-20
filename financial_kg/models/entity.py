"""
Core data models for FinancialKG entities
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np


class Entity(BaseModel):
    """
    Financial entity representation (Company, Person, Sector, Event, etc.)
    """
    id: str = Field(description="Unique entity identifier")
    name: str = Field(description="Entity name")
    type: str = Field(description="Entity type (Company, Person, Sector, Event, Policy, Indicator)")
    label: Optional[str] = Field(default=None, description="Entity label/category")
    
    # Properties specific to entity type
    properties: Dict[str, Any] = Field(default_factory=dict, description="Type-specific properties")
    
    # Embeddings
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    
    # Metadata
    sources: List[str] = Field(default_factory=list, description="Source document/URL IDs")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            datetime: lambda v: v.isoformat()
        }


class CompanyEntity(Entity):
    """Company-specific entity"""
    def __init__(self, **data):
        data['type'] = 'Company'
        super().__init__(**data)
        # Ensure required company properties
        self.properties.setdefault('ticker', None)
        self.properties.setdefault('sector', None)
        self.properties.setdefault('exchange', None)
        self.properties.setdefault('market_cap', None)
        self.properties.setdefault('industry', None)


class PersonEntity(Entity):
    """Person-specific entity (CEO, CFO, Analyst, etc.)"""
    def __init__(self, **data):
        data['type'] = 'Person'
        super().__init__(**data)
        self.properties.setdefault('role', None)
        self.properties.setdefault('affiliation', None)


class SectorEntity(Entity):
    """Sector/Industry entity"""
    def __init__(self, **data):
        data['type'] = 'Sector'
        super().__init__(**data)
        self.properties.setdefault('index', None)
        self.properties.setdefault('industry_group', None)


class EventEntity(Entity):
    """Event entity (Earnings, Merger, Scandal, etc.)"""
    def __init__(self, **data):
        data['type'] = 'Event'
        super().__init__(**data)
        self.properties.setdefault('event_type', None)
        self.properties.setdefault('severity', None)
        self.properties.setdefault('date', None)


class PolicyEntity(Entity):
    """Policy entity (Monetary, Fiscal, Regulatory)"""
    def __init__(self, **data):
        data['type'] = 'Policy'
        super().__init__(**data)
        self.properties.setdefault('policy_type', None)
        self.properties.setdefault('effective_date', None)
        self.properties.setdefault('issuing_authority', None)


class IndicatorEntity(Entity):
    """Economic indicator entity (GDP, CPI, IIP, etc.)"""
    def __init__(self, **data):
        data['type'] = 'Indicator'
        super().__init__(**data)
        self.properties.setdefault('indicator_name', None)
        self.properties.setdefault('value', None)
        self.properties.setdefault('unit', None)
        self.properties.setdefault('period', None)


# Entity factory
def create_entity(entity_type: str, **kwargs) -> Entity:
    """
    Factory function to create the appropriate entity type
    
    Args:
        entity_type: Type of entity to create
        **kwargs: Entity properties
        
    Returns:
        Entity instance of the appropriate type
    """
    entity_map = {
        'Company': CompanyEntity,
        'Person': PersonEntity,
        'Sector': SectorEntity,
        'Event': EventEntity,
        'Policy': PolicyEntity,
        'Indicator': IndicatorEntity,
    }
    
    entity_class = entity_map.get(entity_type, Entity)
    return entity_class(**kwargs)


if __name__ == "__main__":
    # Test entity creation
    company = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries Ltd.",
        label="Energy & Petrochemicals",
        properties={
            "ticker": "RELIANCE",
            "sector": "Energy",
            "exchange": "NSE",
            "market_cap": 1750000000000  # 17.5 trillion INR
        }
    )
    print(f"Created company entity: {company.name}")
    print(f"Properties: {company.properties}")
