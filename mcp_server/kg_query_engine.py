"""
Neo4j-based Knowledge Graph Query Engine for MCP Service

A minimal interface for querying the Neo4j knowledge graph using Cypher queries.
Focuses on security, parameterization, and query control.

Uses official neo4j-python-driver with async API for true concurrent execution.
"""

import time
import re
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from neo4j import AsyncDriver
from lightrag.utils import logger


@dataclass
class EntityInfo:
    """Entity information container"""
    entity_name: str
    description: str
    labels: List[str]
    file_path: str
    created_at: Optional[str] = None


class KGQueryEngine:
    """
    Neo4j-based Knowledge Graph Query Engine

    A secure interface for MCP service using Neo4j Cypher queries with:
    - Parameterized queries for SQL injection prevention
    - Query validation for read-only operations
    - Automatic LIMIT and timeout enforcement
    - Flexible Cypher query execution
    """

    def __init__(
        self,
        driver: AsyncDriver,
        default_limit: int = 30,
        embedding_func=None,
    ):
        """
        Initialize the Neo4j KG Query Engine

        Args:
            driver: Neo4j AsyncDriver instance
            default_limit: Default LIMIT for queries (max 30)
            embedding_func: Embedding function for generating query vectors
        """
        self.driver = driver
        self.default_limit = default_limit
        self.embedding_func = embedding_func
        self._forbidden_keywords = [
            'CREATE', 'DELETE', 'SET', 'REMOVE',
            'MERGE', 'DROP', 'ALTER', 'START', 'FOREACH',
            'CREATE CONSTRAINT', 'CREATE INDEX',
            'DROP CONSTRAINT', 'DROP INDEX', 'DETACH DELETE'
        ]
        self._readonly_patterns = [
            r'\bMATCH\b',
            r'\bRETURN\b',
            r'\bWHERE\b',
            r'\bOPTIONAL MATCH\b',
            r'\bWITH\b',
            r'\bORDER BY\b',
            r'\bSKIP\b',
            r'\bLIMIT\b',
            r'\bCOUNT\b',
            r'\bDISTINCT\b',
            r'\bUNION\b',
            r'\bCALL\b',
            r'\bYIELD\b',
        ]

    def _filter_embedding_fields(self, data: Any) -> Any:
        """
        过滤掉 embedding 相关字段，防止 token 浪费和上下文超限

        Args:
            data: 单个数据字典或数据字典列表

        Returns:
            过滤后的数据
        """
        # 定义需要过滤的字段名模式
        embedding_field_patterns = [
            'embedding',
            'vector',
            'embedding_vector',
            'emb',
            'vec',
            'embedding_array',
            'embedding_list',
        ]

        def filter_single_item(item):
            """过滤单个数据项"""
            # 处理 Neo4j Node 对象或字典
            if hasattr(item, 'keys') and hasattr(item, 'values'):
                item = dict(item)

            if not isinstance(item, dict):
                return item

            filtered_item = {}
            for key, value in item.items():
                # 检查字段名是否匹配 embedding 模式
                is_embedding_field = any(
                    pattern.lower() in key.lower() for pattern in embedding_field_patterns
                )

                if not is_embedding_field:
                    # 递归处理嵌套的值
                    filtered_item[key] = filter_single_item(value)
                else:
                    logger.debug(f"Filtered out embedding field: {key}")

            return filtered_item

        # 处理单个数据项或列表
        if isinstance(data, list):
            return [filter_single_item(item) for item in data]
        else:
            return filter_single_item(data)

    def validate_cypher_query(self, query: str) -> Tuple[bool, str]:
        """
        Validate Cypher query for security (read-only operations)

        Args:
            query: Cypher query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Normalize whitespace and convert to uppercase for keyword checking
        query_normalized = re.sub(r'\s+', ' ', query.strip().upper())

        # Check for forbidden keywords (use word boundaries to avoid partial matches)
        for keyword in self._forbidden_keywords:
            # Use word boundaries to match whole words only
            if re.search(r'\b' + keyword + r'\b', query_normalized):
                return False, f"Forbidden keyword '{keyword}' detected in query"

        # Check if query contains at least one read-only pattern
        has_readonly = any(re.search(pattern, query, re.IGNORECASE)
                          for pattern in self._readonly_patterns)

        if not has_readonly:
            return False, "Query must contain read-only operations (MATCH, RETURN, etc.)"

        # Check for proper syntax
        query_upper = query.upper()
        if 'RETURN' not in query_upper and 'WITH' not in query_upper and 'YIELD' not in query_upper:
            return False, "Query must have RETURN, WITH, or YIELD clause"

        return True, ""

    def apply_query_limits(self, query: str, limit: Optional[int] = None) -> str:
        """
        Apply LIMIT to query if not already present

        Args:
            query: Original Cypher query
            limit: Limit to apply

        Returns:
            Query with LIMIT applied
        """
        if limit is None:
            limit = self.default_limit

        # Remove existing LIMIT clause if present
        query_cleaned = re.sub(r'\s*LIMIT\s+\d+\s*$', '', query.strip(), flags=re.IGNORECASE)

        # Add LIMIT
        return f"{query_cleaned.rstrip(';')} LIMIT {limit}"

    async def vector_search(
        self,
        query_text: str,
        top_k: int = 5,
        node_type: str = "Entity"
    ) -> List[Dict[str, Any]]:
        """
        使用 Neo4j Vector Index 进行向量相似度搜索

        Args:
            query_text: 查询文本
            top_k: 返回数量
            node_type: 节点类型 ("Entity" 或其他)

        Returns:
            相似实体列表
        """
        if self.embedding_func is None:
            raise ValueError("Embedding function not provided, cannot perform vector search")

        # 生成查询向量并转换为list
        query_vector = (await self.embedding_func([query_text]))[0]
        # 确保向量是Python list而不是numpy数组
        if hasattr(query_vector, 'tolist'):
            query_vector = query_vector.tolist()
        elif hasattr(query_vector, 'numpy'):
            query_vector = query_vector.numpy().tolist()

        # 执行向量搜索
        cypher = f"""
        CALL db.index.vector.queryNodes(
            'entity_embedding_index',
            $top_k,
            $query_vector
        ) YIELD node, score
        RETURN
            node.entity_id as entity_name,
            node.description as description,
            [label IN labels(node) WHERE NOT label = 'Entity'] as labels,
            node.file_path as file_path,
            node.created_at as created_at,
            score as similarity_score
        ORDER BY score DESC
        """

        results = await self.execute_cypher(
            cypher,
            {
                "query_vector": query_vector,
                "top_k": top_k
            },
            limit=top_k
        )

        return results

    async def _text_search_fallback(
        self,
        entity_query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        文本搜索备用方案（当向量搜索不可用时）

        Args:
            entity_query: 查询字符串
            top_k: 返回数量

        Returns:
            匹配的实体列表
        """
        cypher = """
        MATCH (e:Entity)
        WHERE e.entity_id CONTAINS $query
           OR e.description CONTAINS $query
           OR any(label IN labels(e) WHERE label CONTAINS $query)
        RETURN
            e.entity_id as entity_name,
            e.description as description,
            [label IN labels(e) WHERE NOT label = 'Entity'] as labels,
            e.file_path as file_path,
            e.created_at as created_at,
            0.0 as similarity_score
        ORDER BY
            CASE
                WHEN e.entity_id = $query THEN 0
                WHEN e.entity_id CONTAINS $query THEN 1
                ELSE 2
            END,
            e.created_at DESC
        """

        results = await self.execute_cypher(
            cypher,
            {"query": entity_query},
            limit=top_k
        )

        return results

    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        vector_params: Optional[Dict[str, bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a parameterized Cypher query with security checks and vector parameter support

        Args:
            query: Cypher query string (can contain $param placeholders)
            parameters: Dictionary of query parameters
            limit: Maximum number of results to return (max 30)
            vector_params: Dict mapping parameter names to boolean flags indicating
                          whether to convert the parameter to a vector (True) or keep as is (False)

        Returns:
            List of result dictionaries

        Raises:
            ValueError: If query is invalid or unsafe
            Exception: If query execution fails
        """
        if limit is None or limit <= 0 or limit > self.default_limit:
            limit = self.default_limit

        # Validate query
        is_valid, error_msg = self.validate_cypher_query(query)
        if not is_valid:
            raise ValueError(f"Query validation failed: {error_msg}")

        # Apply default parameters
        if parameters is None:
            parameters = {}
        else:
            parameters = parameters.copy()  # Create a copy to avoid modifying the original

        # Convert vector parameters if needed
        if vector_params and self.embedding_func:
            for param_name, use_vector in vector_params.items():
                if param_name in parameters and use_vector:
                    param_value = parameters[param_name]
                    if isinstance(param_value, str):
                        # Convert string to vector
                        vector = (await self.embedding_func([param_value]))[0]
                        # 确保向量是Python list而不是numpy数组
                        if hasattr(vector, 'tolist'):
                            vector = vector.tolist()
                        elif hasattr(vector, 'numpy'):
                            vector = vector.numpy().tolist()
                        parameters[param_name] = vector
                        logger.debug(f"Converted parameter '{param_name}' to vector")

        # Apply limit
        if limit is not None:
            query = self.apply_query_limits(query, limit)

        start_time = time.time()

        try:
            logger.info(f"Executing Cypher query (limit={limit or self.default_limit})")
            logger.debug(f"Query: {query}")
            logger.debug(f"Parameters: {parameters}")

            # Execute query using async session
            async with self.driver.session() as session:
                result = await session.run(query, parameters)

                # Convert result to list of dictionaries and filter embedding fields
                results = []
                async for record in result:
                    record_dict = {}
                    for key in record.keys():
                        value = record[key]
                        # Convert Neo4j Node objects to Python dict
                        if hasattr(value, 'items'):
                            node_dict = dict(value)
                            record_dict[key] = node_dict
                        else:
                            record_dict[key] = value

                    # Filter out embedding-related fields
                    record_dict = self._filter_embedding_fields(record_dict)
                    results.append(record_dict)

            elapsed = time.time() - start_time
            logger.info(f"Query executed successfully in {elapsed:.2f}s, returned {len(results)} records")

            return results

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Query execution failed after {elapsed:.2f}s: {e}")
            raise

    async def find_similar_entities(
        self,
        entity_query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar entities using vector similarity search

        Args:
            entity_query: Entity name or description to search for
            top_k: Number of top similar entities to return

        Returns:
            List of dictionaries with similar entity information
        """
        start_time = time.time()

        logger.info(
            f"Searching for entities similar to '{entity_query}' (top_k={top_k})"
        )

        try:
            # 首先尝试向量搜索
            if self.embedding_func is not None:
                try:
                    results = await self.vector_search(entity_query, top_k)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Found {len(results)} similar entities via vector search in {elapsed:.2f}s"
                    )
                    return results
                except Exception as vector_error:
                    logger.warning(f"Vector search failed: {vector_error}, falling back to text search")
                    # 向量搜索失败，使用文本搜索备用方案
                    results = await self._text_search_fallback(entity_query, top_k)
                    elapsed = time.time() - start_time
                    logger.info(
                        f"Found {len(results)} similar entities via text search fallback in {elapsed:.2f}s"
                    )
                    return results
            else:
                # 没有嵌入函数，直接使用文本搜索
                results = await self._text_search_fallback(entity_query, top_k)
                elapsed = time.time() - start_time
                logger.info(
                    f"Found {len(results)} similar entities via text search in {elapsed:.2f}s"
                )
                return results

        except Exception as e:
            logger.error(f"Error searching for similar entities: {e}")
            raise

    async def get_entity_by_name(
        self,
        entity_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific entity by name

        Args:
            entity_name: Name of the entity to retrieve

        Returns:
            Dictionary with entity information or None if not found
        """
        try:
            cypher = """
            MATCH (e:Entity {entity_id: $entity_name})
            RETURN
                e.entity_id as entity_name,
                e.description as description,
                [label IN labels(e) WHERE NOT label = 'Entity'] as labels,
                e.file_path as file_path,
                e.created_at as created_at
            LIMIT 1
            """

            results = await self.execute_cypher(
                cypher,
                {"entity_name": entity_name},
                limit=1
            )

            if not results:
                logger.warning(f"Entity '{entity_name}' not found")
                return None

            # Return first (and only) result
            return results[0]

        except Exception as e:
            logger.error(f"Error retrieving entity '{entity_name}': {e}")
            raise

    async def get_entity_relationships(
        self,
        entity_name: str,
        relationship_type: Optional[str] = None,
        direction: str = "OUTGOING",  # "OUTGOING", "INCOMING", "BOTH"
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get relationships for a specific entity

        Args:
            entity_name: Name of the entity
            relationship_type: Optional relationship type filter
            direction: Direction of relationships (OUTGOING, INCOMING, BOTH)
            limit: Maximum number of relationships to return

        Returns:
            List of relationship information
        """
        try:
            # Build relationship type clause
            if relationship_type:
                rel_clause = f":{relationship_type}"
            else:
                rel_clause = ""

            # Build Cypher query based on direction
            if direction.upper() == "OUTGOING":
                cypher = f"""
                MATCH (e:Entity {{entity_id: $entity_name}})-[r{rel_clause}]->(target:Entity)
                RETURN
                    e.entity_id as source_entity,
                    type(r) as relationship_type,
                    r.description as description,
                    r.weight as weight,
                    target.entity_id as target_entity,
                    target.description as target_description
                ORDER BY r.weight DESC
                """
            elif direction.upper() == "INCOMING":
                cypher = f"""
                MATCH (e:Entity {{entity_id: $entity_name}})<-[r{rel_clause}]-(target:Entity)
                RETURN
                    target.entity_id as source_entity,
                    type(r) as relationship_type,
                    r.description as description,
                    r.weight as weight,
                    e.entity_id as target_entity,
                    e.description as target_description
                ORDER BY r.weight DESC
                """
            else:  # BOTH - 使用 UNION 获取双向关系
                cypher = f"""
                MATCH (e:Entity {{entity_id: $entity_name}})-[r{rel_clause}]->(target:Entity)
                RETURN
                    e.entity_id as source_entity,
                    type(r) as relationship_type,
                    r.description as description,
                    r.weight as weight,
                    target.entity_id as target_entity,
                    target.description as target_description
                UNION
                MATCH (e:Entity {{entity_id: $entity_name}})<-[r{rel_clause}]-(source:Entity)
                RETURN
                    source.entity_id as source_entity,
                    type(r) as relationship_type,
                    r.description as description,
                    r.weight as weight,
                    e.entity_id as target_entity,
                    e.description as target_description
                ORDER BY weight DESC
                """

            results = await self.execute_cypher(
                cypher,
                {"entity_name": entity_name},
                limit=limit
            )

            logger.info(f"Found {len(results)} relationships for entity '{entity_name}'")
            return results

        except Exception as e:
            logger.error(f"Error retrieving relationships for '{entity_name}': {e}")
            raise

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Neo4j connection health

        Returns:
            Dictionary with health status information
        """
        try:
            # Test connection using verify_connectivity
            await self.driver.verify_connectivity()

            return {
                "status": "healthy",
                "neo4j_connected": True,
                "default_limit": self.default_limit,
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "neo4j_connected": False,
                "error": str(e),
            }