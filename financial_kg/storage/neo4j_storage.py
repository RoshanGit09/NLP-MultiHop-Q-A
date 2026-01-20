"""
Neo4j storage backend for FinancialKG
Handles uploading and querying knowledge graphs in Neo4j database
"""
from typing import Optional, List, Dict, Any
from neo4j import GraphDatabase
from datetime import datetime

from models.knowledge_graph import KnowledgeGraph
from models.entity import Entity
from models.relationship import Relationship
from utils.logging_config import get_logger
from utils.config import get_config

logger = get_logger(__name__)
config = get_config()


class Neo4jStorage:
    """
    Neo4j storage backend for Financial Knowledge Graphs
    
    Features:
    - Upload entities as nodes
    - Upload relationships as edges
    - Query and visualization support
    - Cypher query execution
    - Clear/delete operations
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None
    ):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j URI (default from config)
            username: Neo4j username (default from config)
            password: Neo4j password (default from config)
        """
        self.uri = uri or config.neo4j.uri
        self.username = username or config.neo4j.username
        self.password = password or config.neo4j.password
        
        logger.info(f"Connecting to Neo4j at {self.uri}")
        
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("✓ Connected to Neo4j successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def clear_database(self):
        """Clear all nodes and relationships from database"""
        logger.warning("Clearing entire Neo4j database!")
        
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        
        logger.info("✓ Database cleared")
    
    def visualize_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        clear_existing: bool = False
    ):
        """
        Upload knowledge graph to Neo4j for visualization
        
        Args:
            knowledge_graph: KnowledgeGraph to upload
            clear_existing: If True, clear database before upload
        """
        logger.info(f"Uploading KG to Neo4j: {len(knowledge_graph.entities)} entities, "
                   f"{len(knowledge_graph.relationships)} relationships")
        
        if clear_existing:
            self.clear_database()
        
        # Upload entities
        self._upload_entities(knowledge_graph.entities)
        
        # Upload relationships
        self._upload_relationships(knowledge_graph.relationships)
        
        logger.info("✓ Knowledge graph uploaded to Neo4j")
    
    def _upload_entities(self, entities: List[Entity]):
        """Upload entities as nodes"""
        logger.info(f"Uploading {len(entities)} entities as nodes...")
        
        with self.driver.session() as session:
            for entity in entities:
                # Convert entity to node properties
                properties = {
                    'id': entity.id,
                    'name': entity.name,
                    'type': entity.type,
                    'label': entity.label or '',
                    'confidence': entity.confidence,
                    'created_at': entity.created_at.isoformat(),
                    'updated_at': entity.updated_at.isoformat(),
                }
                
                # Add custom properties
                for key, value in entity.properties.items():
                    # Neo4j doesn't support nested dicts, convert to string if needed
                    if isinstance(value, (dict, list)):
                        properties[key] = str(value)
                    elif value is not None:
                        properties[key] = value
                
                # Create node with label based on entity type
                cypher = f"""
                MERGE (n:{entity.type} {{id: $id}})
                SET n += $properties
                """
                
                session.run(cypher, id=entity.id, properties=properties)
        
        logger.info(f"✓ Uploaded {len(entities)} entities")
    
    def _upload_relationships(self, relationships: List[Relationship]):
        """Upload relationships as edges"""
        logger.info(f"Uploading {len(relationships)} relationships as edges...")
        
        with self.driver.session() as session:
            for rel in relationships:
                # Build relationship properties
                properties = {
                    'id': rel.id,
                    'name': rel.name,
                    'predicate': rel.predicate,
                    'created_at': rel.created_at.isoformat(),
                    'updated_at': rel.updated_at.isoformat(),
                }
                
                # Add relationship properties
                if rel.properties.sentiment is not None:
                    properties['sentiment'] = rel.properties.sentiment
                    properties['sentiment_label'] = rel.properties.sentiment_label or ''
                
                if rel.properties.confidence is not None:
                    properties['confidence'] = rel.properties.confidence
                
                if rel.properties.impact_magnitude is not None:
                    properties['impact_magnitude'] = rel.properties.impact_magnitude
                
                if rel.properties.price_change_percent is not None:
                    properties['price_change_percent'] = rel.properties.price_change_percent
                
                if rel.properties.sources:
                    properties['sources'] = ', '.join(rel.properties.sources)
                
                # Add temporal attributes
                if rel.temporal.t_announce:
                    properties['t_announce'] = rel.temporal.t_announce.isoformat()
                
                if rel.temporal.t_effective:
                    properties['t_effective'] = rel.temporal.t_effective.isoformat()
                
                if rel.temporal.t_observe:
                    properties['t_observe'] = rel.temporal.t_observe.isoformat()
                
                if rel.temporal.market_session:
                    properties['market_session'] = rel.temporal.market_session.value
                
                # Create relationship between nodes
                cypher = f"""
                MATCH (a:{rel.subject.type} {{id: $subject_id}})
                MATCH (b:{rel.object.type} {{id: $object_id}})
                MERGE (a)-[r:{rel.predicate.upper().replace(' ', '_').replace('-', '_')} {{id: $rel_id}}]->(b)
                SET r += $properties
                """
                
                session.run(
                    cypher,
                    subject_id=rel.subject.id,
                    object_id=rel.object.id,
                    rel_id=rel.id,
                    properties=properties
                )
        
        logger.info(f"✓ Uploaded {len(relationships)} relationships")
    
    def execute_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a Cypher query
        
        Args:
            cypher_query: Cypher query string
            parameters: Optional query parameters
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Executing query: {cypher_query[:100]}...")
        
        with self.driver.session() as session:
            result = session.run(cypher_query, parameters or {})
            records = [record.data() for record in result]
        
        logger.info(f"✓ Query returned {len(records)} results")
        return records
    
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict]:
        """Get entity by ID"""
        query = "MATCH (n {id: $id}) RETURN n"
        results = self.execute_query(query, {'id': entity_id})
        return results[0]['n'] if results else None
    
    def get_relationships_for_entity(self, entity_id: str) -> List[Dict]:
        """Get all relationships for an entity"""
        query = """
        MATCH (n {id: $id})-[r]-(m)
        RETURN n, r, m
        """
        return self.execute_query(query, {'id': entity_id})
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict]:
        """Get all entities of a specific type"""
        query = f"MATCH (n:{entity_type}) RETURN n"
        results = self.execute_query(query)
        return [r['n'] for r in results]
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph in Neo4j"""
        with self.driver.session() as session:
            # Count nodes
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()['count']
            
            # Count relationships
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            
            # Node types
            node_types = session.run("""
                MATCH (n)
                RETURN labels(n)[0] as type, count(*) as count
                ORDER BY count DESC
            """).data()
            
            # Relationship types
            rel_types = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(*) as count
                ORDER BY count DESC
            """).data()
        
        return {
            'total_nodes': node_count,
            'total_relationships': rel_count,
            'node_types': node_types,
            'relationship_types': rel_types
        }
    
    def search_entities_by_name(self, name_pattern: str) -> List[Dict]:
        """Search entities by name pattern"""
        query = """
        MATCH (n)
        WHERE n.name CONTAINS $pattern
        RETURN n
        LIMIT 100
        """
        results = self.execute_query(query, {'pattern': name_pattern})
        return [r['n'] for r in results]
    
    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Get distribution of sentiment labels across relationships"""
        query = """
        MATCH ()-[r]->()
        WHERE r.sentiment_label IS NOT NULL
        RETURN r.sentiment_label as label, count(*) as count
        ORDER BY count DESC
        """
        results = self.execute_query(query)
        return {r['label']: r['count'] for r in results}


if __name__ == "__main__":
    # Test Neo4j storage
    from models.entity import create_entity
    from models.relationship import Relationship, RelationshipProperties
    from models.temporal_models import TemporalAttributes
    
    print("Testing Neo4j Storage...")
    
    # Create a simple KG
    kg = KnowledgeGraph()
    
    # Add entities
    reliance = create_entity(
        entity_type="Company",
        id="RELIANCE",
        name="Reliance Industries",
        properties={"ticker": "RELIANCE", "sector": "Energy"}
    )
    
    energy = create_entity(
        entity_type="Sector",
        id="ENERGY",
        name="Energy Sector",
        properties={"index": "NIFTY_ENERGY"}
    )
    
    kg.add_entity(reliance)
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
            sentiment_label="positive",
            confidence=0.95
        ),
        temporal=TemporalAttributes(
            t_observe=datetime.now()
        )
    )
    
    kg.add_relationship(rel)
    
    # Upload to Neo4j
    try:
        with Neo4jStorage() as storage:
            print("\n1. Uploading to Neo4j...")
            storage.visualize_graph(kg, clear_existing=True)
            
            print("\n2. Getting stats...")
            stats = storage.get_graph_stats()
            print(f"  Nodes: {stats['total_nodes']}")
            print(f"  Relationships: {stats['total_relationships']}")
            
            print("\n3. Searching for Reliance...")
            results = storage.search_entities_by_name("Reliance")
            print(f"  Found: {len(results)} entities")
            
            print("\n✓ Neo4j storage test complete!")
            print("\nOpen Neo4j Browser at http://localhost:7474")
            print("Run: MATCH (n) RETURN n")
    except Exception as e:
        print(f"\n✗ Neo4j test failed: {e}")
        print("\nMake sure Neo4j is running!")
        print("Install: Download from https://neo4j.com/download/")
        print("Or use Neo4j Desktop or Docker")
