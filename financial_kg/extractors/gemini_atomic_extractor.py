"""
Gemini-powered atomic fact extractor for financial knowledge graph construction
"""
import asyncio
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

from ..utils.logging_config import get_logger
from ..utils.gemini_client import get_gemini_client
from ..models.entity import Entity, create_entity
from ..models.relationship import Relationship, RelationshipProperties
from ..models.temporal_models import TemporalAttributes

logger = get_logger(__name__)


class GeminiAtomicExtractor:
    """
    Gemini-powered extractor for atomic facts from financial text
    
    Extends the ATOM methodology for financial domain, extracting:
    - Entities (Companies, People, Events, Indicators)
    - Relationships with financial sentiment
    - Temporal attributes (4-dimensional time model)
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize Gemini extractor.
        Uses GEMINI_MODEL_FAST from .env (default: gemini-2.5-flash).
        Pass model_name explicitly only to override.
        """
        self.client = get_gemini_client()
        # Use the configured fast model from .env, not a hardcoded legacy value
        self.model_name = model_name or self.client.model_fast_name
        logger.info(f"GeminiAtomicExtractor initialized with model: {self.model_name}")
    
    async def extract_atomic_facts(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Extract atomic facts from financial text
        
        Args:
            text: Input financial text (news, earnings calls, etc.)
            context: Additional context (date, source, ticker, etc.)
            
        Returns:
            List of atomic fact dictionaries
        """
        prompt = self._create_extraction_prompt(text, context)
        
        try:
            # Use the model resolved at __init__ time (respects GEMINI_MODEL_FAST from .env)
            response_text = await self.client.generate_async(
                prompt,
                use_pro=(self.model_name == self.client.model_pro_name),
                temperature=0.0
            )
            
            # Log the raw response for debugging
            logger.debug(f"Raw Gemini response: {response_text[:500]}")
            
            # Clean response - remove markdown code blocks if present
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith("```"):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith("```"):
                response_text = response_text[:-3]  # Remove trailing ```
            response_text = response_text.strip()
            
            # Parse JSON response
            facts = json.loads(response_text)
            logger.info(f"Extracted {len(facts)} atomic facts from text")
            
            return facts
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error: {e}")
            logger.error(f"Response text: {response_text[:1000]}")
            return []
        except Exception as e:
            logger.error(f"Extraction failed: {e}", exc_info=True)
            return []
    
    def _create_extraction_prompt(self, text: str, context: Optional[Dict] = None) -> str:
        """Create prompt for atomic fact extraction"""
        
        base_prompt = """
You are a financial knowledge graph expert. Extract atomic facts from the given financial text.

For each atomic fact, return a JSON object with:
{
  "subject": "entity name",
  "subject_type": "Company|Person|Sector|Event|Policy|Indicator",
  "predicate": "relationship type",
  "object": "entity name", 
  "object_type": "Company|Person|Sector|Event|Policy|Indicator",
  "confidence": 0.0-1.0,
  "sentiment": -1.0-1.0,
  "temporal": {
    "t_announce": "when announced (YYYY-MM-DD)",
    "t_effective": "when takes effect (YYYY-MM-DD)", 
    "t_observe": "when observed (YYYY-MM-DD)",
    "t_impact": "when impact felt (YYYY-MM-DD)"
  },
  "source_text": "exact text span",
  "properties": {
    // Additional entity-specific properties
  }
}

Financial entity types:
- Company: Stock ticker, sector, exchange, market cap
- Person: Role, affiliation, title
- Sector: Industry group, index (NIFTY, SENSEX)
- Event: Earnings, mergers, policy announcements
- Policy: RBI policies, budget announcements
- Indicator: Stock prices, financial ratios, economic indicators

Relationship types:
- OPERATES_IN, COMPETES_WITH, SUPPLIES_TO
- ANNOUNCES, REPORTS, INVESTS_IN
- IMPACTS, AFFECTS, CORRELATES_WITH
- LEADS, MANAGES, OWNS

Return only valid JSON array of facts. No explanation.

"""
        
        if context:
            base_prompt += f"\\nContext: {json.dumps(context, default=str)}"
        
        base_prompt += f"\\nText to analyze:\\n{text}"
        
        return base_prompt
    
    async def extract_entities_and_relationships(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text
        
        Args:
            text: Input financial text
            context: Additional context (date, source, etc.)
            
        Returns:
            Tuple of (entities, relationships)
        """
        # First extract atomic facts
        facts = await self.extract_atomic_facts(text, context)
        
        # Then convert to entities and relationships
        return self._convert_facts_to_entities_relationships(facts)
    
    def _convert_facts_to_entities_relationships(
        self,
        facts: List[Dict]
    ) -> tuple[List[Entity], List[Relationship]]:
        """
        Convert atomic facts to entities and relationships
        
        Args:
            facts: List of atomic fact dictionaries
            
        Returns:
            Tuple of (entities, relationships)
        """
        entities = []
        relationships = []
        entity_cache = {}  # normalized key -> Entity (deduplicates within this call)
        
        for fact in facts:
            try:
                # Skip if fact is not a dict
                if not isinstance(fact, dict):
                    logger.warning(f"Skipping non-dict fact: {type(fact)}")
                    continue
                
                # Extract subject entity
                subject = self._create_entity_from_fact(fact, "subject", entity_cache)
                if subject and subject.id not in {e.id for e in entities}:
                    entities.append(subject)
                
                # Extract object entity  
                object_entity = self._create_entity_from_fact(fact, "object", entity_cache)
                if object_entity and object_entity.id not in {e.id for e in entities}:
                    entities.append(object_entity)
                
                # Create relationship
                if subject and object_entity:
                    relationship = self._create_relationship_from_fact(fact, subject, object_entity)
                    if relationship:
                        relationships.append(relationship)
                        
            except Exception as e:
                logger.error(f"Failed to process fact: {e}")
                continue
        
        logger.info(f"Created {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    def _create_entity_from_fact(
        self,
        fact: Dict,
        role: str,
        entity_cache: Dict
    ) -> Optional[Entity]:
        """Create entity from atomic fact"""

        entity_name = fact.get(role)
        entity_type = fact.get(f"{role}_type")

        # Coerce name to string — Gemini sometimes returns numbers
        if entity_name is None:
            return None
        entity_name = str(entity_name).strip()
        if not entity_name or entity_name.upper() in ("N/A", "NA", "NONE", "NULL", "UNKNOWN"):
            return None

        # Default type when Gemini omits it
        if not entity_type or not isinstance(entity_type, str):
            entity_type = "Indicator"
        else:
            entity_type = entity_type.strip()
            _VALID_TYPES = {"Company", "Person", "Sector", "Event", "Policy", "Indicator"}
            if entity_type not in _VALID_TYPES:
                entity_type = "Indicator"

        # Normalize for cache key (case + whitespace insensitive)
        cache_key = f"{entity_type.lower()}:{entity_name.lower()}"
        if cache_key in entity_cache:
            return entity_cache[cache_key]

        # Normalize entity ID (consistent casing/separators)
        norm_id = entity_name.upper().replace(" ", "_").replace("-", "_")

        # Create entity
        properties = fact.get("properties") or {}
        if not isinstance(properties, dict):
            properties = {}
        entity = create_entity(
            entity_type=entity_type,
            id=norm_id,
            name=entity_name,
            properties=properties
        )

        entity_cache[cache_key] = entity
        return entity
    
    def _create_relationship_from_fact(
        self,
        fact: Dict,
        subject: Entity,
        object_entity: Entity
    ) -> Optional[Relationship]:
        """Create relationship from atomic fact"""
        
        predicate = fact.get("predicate")
        if not predicate:
            return None
        
        # Create temporal attributes
        temporal_data = fact.get("temporal", {})
        temporal = TemporalAttributes(
            t_announce=temporal_data.get("t_announce"),
            t_effective=temporal_data.get("t_effective"),
            t_observe=temporal_data.get("t_observe"),
            t_impact_start=temporal_data.get("t_impact") or temporal_data.get("t_impact_start"),
            t_impact_end=temporal_data.get("t_impact_end")
        )
        
        # Create relationship properties
        properties = RelationshipProperties(
            confidence=fact.get("confidence", 0.8),
            sentiment=fact.get("sentiment", 0.0),
            source_text=fact.get("source_text", ""),
            temporal=temporal
        )
        
        # Generate unique relationship ID and name
        relationship_id = f"{subject.id}_{predicate.upper().replace(' ', '_')}_{object_entity.id}"
        
        return Relationship(
            id=relationship_id,
            name=predicate,
            subject=subject,
            predicate=predicate,
            object=object_entity,
            properties=properties
        )
