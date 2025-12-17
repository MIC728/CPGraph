"""
Extended query interfaces for dual-dimension entity type filtering in LightRAG.

This module provides enhanced query capabilities that support filtering entities
by both dimension1 (technical classification) and dimension2 (application level)
entity types.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

from lightrag.base import BaseGraphStorage, BaseVectorStorage
from lightrag.utils import logger


class DualDimensionQueryInterface:
    """Enhanced query interface supporting dual-dimension entity type filtering."""
    
    def __init__(
        self,
        knowledge_graph_inst: BaseGraphStorage,
        entities_vdb: BaseVectorStorage,
        relationships_vdb: BaseVectorStorage,
    ):
        """
        Initialize the dual-dimension query interface.
        
        Args:
            knowledge_graph_inst: Knowledge graph storage instance
            entities_vdb: Entity vector database
            relationships_vdb: Relationship vector database
        """
        self.knowledge_graph_inst = knowledge_graph_inst
        self.entities_vdb = entities_vdb
        self.relationships_vdb = relationships_vdb
    
    async def query_entities_by_dimensions(
        self,
        query: str,
        dim1_types: Optional[List[str]] = None,
        dim2_types: Optional[List[str]] = None,
        top_k: int = 40,
        min_cosine_score: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Query entities filtered by dual-dimension types.
        
        Args:
            query: Query string
            dim1_types: List of dimension1 types to filter by (technical classification)
            dim2_types: List of dimension2 types to filter by (application level)
            top_k: Number of top results to return
            min_cosine_score: Minimum cosine similarity score
            
        Returns:
            List of entity dictionaries with dual-dimension type information
        """
        logger.info(f"Querying entities with dim1_types={dim1_types}, dim2_types={dim2_types}")
        
        # Step 1: Get all entities via vector search
        results = await self.entities_vdb.query(query, top_k=top_k * 2)  # Get more results for filtering
        
        if not results:
            return []
        
        # Step 2: Filter entities by dual-dimension types
        filtered_entities = []
        for result in results:
            if result.get("cosine_similarity", 1.0) < min_cosine_score:
                continue
                
            entity_name = result.get("entity_name")
            if not entity_name:
                continue
            
            # Get full entity data from knowledge graph
            entity_data = await self.knowledge_graph_inst.get_node(entity_name)
            if not entity_data:
                continue
            
            # Check dimension1 type filter
            if dim1_types:
                entity_dim1 = entity_data.get("entity_type_dim1", "")
                if entity_dim1 not in dim1_types:
                    continue
            
            # Check dimension2 type filter
            if dim2_types:
                entity_dim2 = entity_data.get("entity_type_dim2", "")
                if entity_dim2 not in dim2_types:
                    continue
            
            # Add cosine similarity from vector search result
            entity_data["cosine_similarity"] = result.get("cosine_similarity", 1.0)
            filtered_entities.append(entity_data)
        
        # Sort by cosine similarity and return top_k
        filtered_entities.sort(key=lambda x: x.get("cosine_similarity", 0), reverse=True)
        return filtered_entities[:top_k]
    
    async def get_entities_by_type_combinations(
        self,
        dim1_types: Optional[List[str]] = None,
        dim2_types: Optional[List[str]] = None,
        limit: int = 100,
    ) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
        """
        Get entities grouped by dual-dimension type combinations.
        
        Args:
            dim1_types: List of dimension1 types to include
            dim2_types: List of dimension2 types to include
            limit: Maximum number of entities per type combination
            
        Returns:
            Dictionary mapping (dim1_type, dim2_type) tuples to lists of entity data
        """
        logger.info(f"Getting entities by type combinations: dim1={dim1_types}, dim2={dim2_types}")
        
        # Get all nodes from knowledge graph
        all_nodes = await self.knowledge_graph_inst.get_all_nodes()
        
        # Group entities by type combinations
        type_combinations = defaultdict(list)
        
        for node in all_nodes:
            entity_dim1 = node.get("entity_type_dim1", "")
            entity_dim2 = node.get("entity_type_dim2", "")
            
            # Apply filters
            if dim1_types and entity_dim1 not in dim1_types:
                continue
            if dim2_types and entity_dim2 not in dim2_types:
                continue
            
            type_key = (entity_dim1, entity_dim2)
            if len(type_combinations[type_key]) < limit:
                type_combinations[type_key].append(node)
        
        return dict(type_combinations)
    
    async def get_type_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about entity type distributions.
        
        Returns:
            Dictionary containing type distribution statistics
        """
        # Get all nodes from knowledge graph
        all_nodes = await self.knowledge_graph_inst.get_all_nodes()
        
        dim1_counts = Counter()
        dim2_counts = Counter()
        combination_counts = Counter()
        
        for node in all_nodes:
            entity_dim1 = node.get("entity_type_dim1", "")
            entity_dim2 = node.get("entity_type_dim2", "")
            
            dim1_counts[entity_dim1] += 1
            dim2_counts[entity_dim2] += 1
            combination_counts[(entity_dim1, entity_dim2)] += 1
        
        return {
            "total_entities": len(all_nodes),
            "dim1_distribution": dict(dim1_counts),
            "dim2_distribution": dict(dim2_counts),
            "combination_distribution": {
                f"{dim1}+{dim2}": count 
                for (dim1, dim2), count in combination_counts.items()
            },
            "unique_dim1_types": len(dim1_counts),
            "unique_dim2_types": len(dim2_counts),
            "unique_combinations": len(combination_counts),
        }
    
    async def find_related_entities_by_type(
        self,
        entity_name: str,
        target_dim1_types: Optional[List[str]] = None,
        target_dim2_types: Optional[List[str]] = None,
        max_hops: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find entities related to a given entity, filtered by target types.
        
        Args:
            entity_name: Name of the source entity
            target_dim1_types: List of target dimension1 types to filter by
            target_dim2_types: List of target dimension2 types to filter by
            max_hops: Maximum number of hops in the knowledge graph
            
        Returns:
            List of related entity dictionaries
        """
        logger.info(f"Finding entities related to '{entity_name}' with type filters")
        
        # Check if source entity exists
        source_entity = await self.knowledge_graph_inst.get_node(entity_name)
        if not source_entity:
            logger.warning(f"Source entity '{entity_name}' not found")
            return []
        
        # Get edges connected to the source entity
        connected_edges = await self.knowledge_graph_inst.get_node_edges(entity_name)
        if not connected_edges:
            return []
        
        # Collect related entities
        related_entities = []
        seen_entities = set()
        
        for src_id, tgt_id in connected_edges:
            # Determine the target entity
            target_entity_name = tgt_id if src_id == entity_name else src_id
            
            if target_entity_name in seen_entities:
                continue
            seen_entities.add(target_entity_name)
            
            # Get target entity data
            target_entity = await self.knowledge_graph_inst.get_node(target_entity_name)
            if not target_entity:
                continue
            
            # Apply type filters
            if target_dim1_types:
                target_dim1 = target_entity.get("entity_type_dim1", "")
                if target_dim1 not in target_dim1_types:
                    continue
            
            if target_dim2_types:
                target_dim2 = target_entity.get("entity_type_dim2", "")
                if target_dim2 not in target_dim2_types:
                    continue
            
            # Add relationship information
            edge_data = await self.knowledge_graph_inst.get_edge(
                entity_name, target_entity_name
            )
            target_entity["relationship"] = edge_data
            target_entity["is_source"] = (src_id == entity_name)
            
            related_entities.append(target_entity)
        
        return related_entities
    
    async def advanced_dual_dimension_search(
        self,
        query: str,
        dim1_filters: Dict[str, float],  # type -> weight
        dim2_filters: Dict[str, float],  # type -> weight
        top_k: int = 40,
        combine_strategy: str = "weighted_sum",
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with weighted dual-dimension filters.
        
        Args:
            query: Query string
            dim1_filters: Dictionary of dim1 types and their weights
            dim2_filters: Dictionary of dim2 types and their weights
            top_k: Number of top results to return
            combine_strategy: How to combine vector similarity with type weights ('weighted_sum', 'multiplicative')
            
        Returns:
            List of ranked entity dictionaries
        """
        logger.info(f"Advanced dual-dimension search: dim1_filters={dim1_filters}, dim2_filters={dim2_filters}")
        
        # Get initial vector search results
        vector_results = await self.entities_vdb.query(query, top_k=top_k * 2)
        
        if not vector_results:
            return []
        
        # Process and score entities
        scored_entities = []
        for result in vector_results:
            entity_name = result.get("entity_name")
            if not entity_name:
                continue
            
            # Get entity data
            entity_data = await self.knowledge_graph_inst.get_node(entity_name)
            if not entity_data:
                continue
            
            # Calculate vector similarity score
            vector_score = result.get("cosine_similarity", 0.0)
            
            # Calculate type weight scores
            entity_dim1 = entity_data.get("entity_type_dim1", "")
            entity_dim2 = entity_data.get("entity_type_dim2", "")
            
            dim1_weight = dim1_filters.get(entity_dim1, 0.0)
            dim2_weight = dim2_filters.get(entity_dim2, 0.0)
            
            # Combine scores based on strategy
            if combine_strategy == "weighted_sum":
                final_score = vector_score + dim1_weight + dim2_weight
            elif combine_strategy == "multiplicative":
                final_score = vector_score * (1 + dim1_weight) * (1 + dim2_weight)
            else:
                final_score = vector_score
            
            # Create result with scores
            result_entity = entity_data.copy()
            result_entity.update({
                "vector_similarity": vector_score,
                "dim1_weight": dim1_weight,
                "dim2_weight": dim2_weight,
                "final_score": final_score,
                "dim1_type": entity_dim1,
                "dim2_type": entity_dim2,
            })
            
            scored_entities.append(result_entity)
        
        # Sort by final score and return top_k
        scored_entities.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        return scored_entities[:top_k]


# Convenience functions for backward compatibility
async def query_entities_with_dimensions(
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    query: str,
    dim1_types: Optional[List[str]] = None,
    dim2_types: Optional[List[str]] = None,
    top_k: int = 40,
) -> List[Dict[str, Any]]:
    """
    Convenience function for dual-dimension entity querying.
    
    Args:
        knowledge_graph_inst: Knowledge graph storage instance
        entities_vdb: Entity vector database
        query: Query string
        dim1_types: List of dimension1 types to filter by
        dim2_types: List of dimension2 types to filter by
        top_k: Number of top results to return
        
    Returns:
        List of filtered entity dictionaries
    """
    interface = DualDimensionQueryInterface(
        knowledge_graph_inst, entities_vdb, knowledge_graph_inst  # relationships_vdb not used for entity queries
    )
    
    return await interface.query_entities_by_dimensions(
        query=query,
        dim1_types=dim1_types,
        dim2_types=dim2_types,
        top_k=top_k,
    )


async def get_entity_type_statistics(
    knowledge_graph_inst: BaseGraphStorage,
) -> Dict[str, Any]:
    """
    Convenience function for getting entity type statistics.
    
    Args:
        knowledge_graph_inst: Knowledge graph storage instance
        
    Returns:
        Dictionary containing type distribution statistics
    """
    interface = DualDimensionQueryInterface(
        knowledge_graph_inst, None, None  # VDB instances not needed for statistics
    )
    
    return await interface.get_type_statistics()
