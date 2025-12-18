"""
MCP Server for Neo4j Knowledge Graph Query Engine

基于 FastMCP 的知识图谱查询服务，使用 Neo4j 数据库和 Cypher 查询。
使用官方 neo4j-python-driver 异步 API 实现真正的并发查询。
参考 https://gofastmcp.com 了解更多信息。
"""

import asyncio
import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastmcp import FastMCP
from kg_query_engine import KGQueryEngine

# Neo4j 相关导入 - 使用官方异步驱动
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv
import pathlib
from lightrag.utils import setup_logger, get_env_value

# 设置日志 - 复用LightRAG logger
log_level = get_env_value("LOG_LEVEL", "INFO", str).upper()
setup_logger(
    logger_name="CPGraph",
    level=log_level,
    add_filter=False,
    enable_file_logging=False  # Console only
)
logger = logging.getLogger("CPGraph")

# 加载 .env 文件 - 尝试多个可能的路径
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent

for env_path in [project_root / ".env", script_dir / ".env", pathlib.Path(".env")]:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
        logger.info(f"[OK] Load environment variables: {env_path}")
        break

def filter_embedding_fields(data):
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
        if not isinstance(item, dict):
            return item

        filtered_item = {}
        for key, value in item.items():
            # 检查字段名是否匹配 embedding 模式
            is_embedding_field = any(
                pattern.lower() in key.lower() for pattern in embedding_field_patterns
            )

            if not is_embedding_field:
                filtered_item[key] = value
            else:
                logger.debug(f"Filtered out embedding field: {key}")

        return filtered_item

    # 处理单个数据项或列表
    if isinstance(data, list):
        return [filter_single_item(item) for item in data]
    else:
        return filter_single_item(data)


def safe_execute_cypher_with_filter(query_engine, *args, **kwargs):
    """
    安全执行 Cypher 查询，自动过滤 embedding 字段

    Args:
        query_engine: KGQueryEngine 实例
        *args: 传递给 execute_cypher 的位置参数
        **kwargs: 传递给 execute_cypher 的关键字参数

    Returns:
        过滤后的查询结果
    """
    import asyncio

    # 获取事件循环
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # 执行查询
    results = loop.run_until_complete(
        query_engine.execute_cypher(*args, **kwargs)
    )

    # 过滤结果
    filtered_results = filter_embedding_fields(results)

    # 记录过滤统计
    if isinstance(results, list) and len(results) > 0:
        original_size = len(str(results))
        filtered_size = len(str(filtered_results))
        saved_ratio = (original_size - filtered_size) / original_size * 100
        logger.info(f"Filtered embedding fields: saved {saved_ratio:.1f}% of response size")

    return filtered_results



# Initialize FastMCP server
mcp = FastMCP(
    "OI Knowledge Graph Query Server",
    instructions="LightRAG 知识图谱查询服务，专门用于信息学竞赛知识查询"
)

class KGQueryService:
    """Neo4j 知识图谱查询服务核心类 - 使用官方异步驱动"""

    def __init__(self):
        self.driver = None  # AsyncDriver 实例（内置连接池）
        self.kg_engine = None  # 单个引擎实例
        self.initialized = False
        logger.info(f"[OK] KGQueryService initialized")

    async def initialize(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        default_limit: int = 30,
        embedding_func=None,
    ):
        """
        初始化 Neo4j 知识图谱查询引擎

        Args:
            neo4j_uri: Neo4j 连接 URI
            neo4j_user: Neo4j 用户名
            neo4j_password: Neo4j 密码
            default_limit: 默认查询限制 (max 30)
            embedding_func: 嵌入函数，用于向量搜索
        """
        if self.initialized:
            logger.warning("[WARNING] KGQueryEngine already initialized, skip duplicate initialization")
            return

        try:
            # 从环境变量读取 Neo4j 配置
            uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
            user = neo4j_user or os.getenv("NEO4J_USERNAME", "neo4j")
            password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")

            logger.info(f"[INFO] Initializing Neo4j AsyncDriver...")
            logger.info(f"  URI: {uri}")
            logger.info(f"  User: {user}")
            logger.info(f"  Default limit: {default_limit}")
            logger.info(f"  Vector search: {'[ENABLED]' if embedding_func else '[DISABLED]'}")

            # 创建 AsyncDriver（内置连接池管理）
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

            # 验证连接
            await self.driver.verify_connectivity()
            logger.info(f"[OK] Neo4j connection verified")

            # 创建单个引擎实例
            self.kg_engine = KGQueryEngine(
                driver=self.driver,
                default_limit=default_limit,
                embedding_func=embedding_func,
            )
            logger.info(f"[OK] KGQueryEngine initialized")

            self.initialized = True
            logger.info(f"[OK] KGQueryService initialized successfully (async driver with built-in connection pool)")

        except Exception as e:
            logger.error(f"[ERROR] KGQueryEngine initialization failed: {e}")
            raise

    async def cleanup(self):
        """清理资源，关闭驱动连接"""
        if self.driver:
            await self.driver.close()
            logger.info("[OK] Neo4j driver closed")

    async def find_similar_entities(
        self,
        entity_query: str,
        top_k: int = 5,
        rerank: str = "pass",
    ) -> List[Dict[str, Any]]:
        """
        根据查询字符串查找相似的实体

        Args:
            entity_query: 实体查询字符串（名称或描述）
            top_k: 返回的最相似实体数量
            rerank: 重排序策略：
                - "pass": 跳过重排，使用原始向量相似度排序（默认）
                - "degree": 基于子图度数重排序

        Returns:
            相似实体列表，每个实体包含名称、描述、类型等信息。
            当rerank="degree"时，还会包含degree（度数）和similarity_score（相似度分数）用于调试
        """
        if not self.initialized:
            logger.info("服务初始化")
            await self.initialize()

        # 计算实际向量检索的top_k
        vector_top_k = top_k * 4 if rerank == "degree" else top_k
        logger.info(f"[INFO] Finding similar entities: '{entity_query}' (top_k={top_k}, rerank={rerank}, vector_top_k={vector_top_k})")

        try:
            results = await self.kg_engine.find_similar_entities(
                entity_query=entity_query,
                top_k=vector_top_k,
            )

            # 如果需要重排序
            if rerank == "degree" and len(results) > top_k:
                results = await self._rerank_by_degree(results, top_k)
            elif rerank == "degree":
                # 即使没有截断，也计算度数用于返回
                results_with_scores = await self._rerank_by_degree(results, len(results))
                # 恢复原始顺序但保留分数
                for i, r in enumerate(results):
                    if i < len(results_with_scores):
                        r["degree"] = results_with_scores[i].get("degree", 0)
                        # 保留原始的similarity_score（如果存在），或者使用重排序后的值
                        r["similarity_score"] = results_with_scores[i].get("similarity_score", r.get("similarity_score", 0))
            elif rerank == "pagerank" and len(results) > top_k:
                results = await self._rerank_by_pagerank(results, top_k)
            elif rerank == "pagerank":
                # 即使没有截断，也计算 PageRank 用于返回
                results_with_scores = await self._rerank_by_pagerank(results, len(results))
                # 恢复原始顺序但保留分数
                for i, r in enumerate(results):
                    if i < len(results_with_scores):
                        r["pagerank"] = results_with_scores[i].get("pagerank", 0)
                        # 保留原始的similarity_score（如果存在），或者使用重排序后的值
                        r["similarity_score"] = results_with_scores[i].get("similarity_score", r.get("similarity_score", 0))

            # 注意：结果已经在 kg_query_engine.execute_cypher 中自动过滤了 embedding 字段
            logger.info(f"[OK] Found {len(results)} similar entities (embedding fields automatically filtered)")

            return results

        except Exception as e:
            logger.error(f"[ERROR] Failed to find similar entities: {e}")
            raise

    async def _rerank_by_degree(
        self,
        candidates: List[Dict],
        final_k: int,
    ) -> List[Dict]:
        """
        基于子图度数重排序

        Args:
            candidates: 候选实体列表
            final_k: 最终返回的数量

        Returns:
            重排序后的结果，包含degree和similarity_score字段用于调试
        """
        if not candidates:
            return candidates

        # 提取候选节点ID列表
        candidate_ids = [c["entity_name"] for c in candidates]

        # 构建子图并计算度数
        query = """
        UNWIND $candidate_ids as target_id
        MATCH (n:Entity {entity_id: target_id})
        WITH target_id, n

        MATCH (n)-[r]-(neighbor:Entity)
        WHERE neighbor.entity_id IN $candidate_ids
        WITH target_id, count(DISTINCT neighbor) as degree

        RETURN target_id, degree
        ORDER BY degree DESC
        """

        degree_results = await self.kg_engine.execute_cypher(
            query=query,
            parameters={"candidate_ids": candidate_ids}
        )

        # 创建度数映射
        degree_map = {r["target_id"]: r["degree"] for r in degree_results}

        # 重排序：度数主导，相似度辅助
        # 先为每个候选计算rerank_score并排序，保留度数和相似度分数
        reranked_with_scores = []
        for candidate in candidates:
            entity_name = candidate["entity_name"]
            degree = degree_map.get(entity_name, 0)
            # 注意：vector_search返回的是similarity_score字段，不是score字段
            similarity_score = candidate.get("similarity_score", 0) if "similarity_score" in candidate else 0
            rerank_score = degree * 1000 + similarity_score

            # 为候选添加度数和相似度分数字段
            candidate_with_scores = candidate.copy()
            candidate_with_scores["degree"] = degree
            candidate_with_scores["similarity_score"] = similarity_score

            reranked_with_scores.append((candidate_with_scores, rerank_score))

        # 按rerank_score排序
        reranked_with_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回最终结果（包含度数和相似度分数字段）
        final_results = [candidate for candidate, score in reranked_with_scores[:final_k]]

        # 输出调试信息到日志
        logger.info(f"[DEBUG] Reranked {len(candidates)} candidates, top 3 scores:")
        for i, (candidate, score) in enumerate(reranked_with_scores[:3]):
            logger.info(f"  {i+1}. {candidate['entity_name']}: degree={candidate['degree']}, similarity={candidate['similarity_score']:.4f}, rerank_score={score:.4f}")

        return final_results

    async def _rerank_by_pagerank(
        self,
        candidates: List[Dict],
        final_k: int,
    ) -> List[Dict]:
        """
        基于全图 PageRank 重排序

        使用 Neo4j 的 PageRank 算法计算每个候选实体在全局图中的重要性，
        然后结合原始相似度分数进行重排序。

        Args:
            candidates: 候选实体列表
            final_k: 最终返回的数量

        Returns:
            重排序后的结果，包含 pagerank 和 similarity_score 字段
        """
        if not candidates:
            return candidates

        # 提取候选节点ID列表
        candidate_ids = [c["entity_name"] for c in candidates]

        try:
            # 直接查询候选实体的 PageRank 值（已预先计算并存储在节点属性中）
            query = """
            MATCH (e:Entity)
            WHERE e.entity_id IN $candidate_ids
            RETURN e.entity_id as entity_id, e.pagerank as pagerank
            ORDER BY e.pagerank DESC
            """

            pagerank_results = await self.kg_engine.execute_cypher(
                query=query,
                parameters={"candidate_ids": candidate_ids}
            )

            # 创建 PageRank 映射
            pagerank_map = {r["entity_id"]: r["pagerank"] for r in pagerank_results}

            # 重排序：PageRank 主导，相似度辅助
            reranked_with_scores = []
            for candidate in candidates:
                entity_name = candidate["entity_name"]
                pagerank = pagerank_map.get(entity_name, 0)
                similarity_score = candidate.get("similarity_score", 0) if "similarity_score" in candidate else 0
                rerank_score = pagerank * 1000 + similarity_score

                # 为候选添加 PageRank 和相似度分数字段
                candidate_with_scores = candidate.copy()
                candidate_with_scores["pagerank"] = pagerank
                candidate_with_scores["similarity_score"] = similarity_score

                reranked_with_scores.append((candidate_with_scores, rerank_score))

            # 按 rerank_score 排序
            reranked_with_scores.sort(key=lambda x: x[1], reverse=True)

            # 返回最终结果（包含 PageRank 和相似度分数字段）
            final_results = [candidate for candidate, score in reranked_with_scores[:final_k]]

            # 输出调试信息
            logger.info(f"[DEBUG] PageRank reranked {len(candidates)} candidates, top 3 scores:")
            for i, (candidate, score) in enumerate(reranked_with_scores[:3]):
                logger.info(f"  {i+1}. {candidate['entity_name']}: pagerank={candidate['pagerank']:.4f}, similarity={candidate['similarity_score']:.4f}, rerank_score={score:.4f}")

            return final_results

        except Exception as e:
            logger.warning(f"[WARNING] PageRank query failed: {e}, falling back to original order")
            # 如果 PageRank 查询失败，返回原始结果
            return candidates[:final_k]

    async def execute_custom_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        vector_params: Optional[Dict[str, bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        执行自定义 Cypher 查询

        Args:
            query: Cypher 查询字符串
            parameters: 查询参数
            limit: 结果限制 (max 30)
            vector_params: 向量参数映射，标记哪些参数需要转换为向量

        Returns:
            查询结果列表
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"[INFO] Executing custom Cypher query")
        logger.info(f"  Query length: {len(query)} characters")
        logger.info(f"  Parameters count: {len(parameters) if parameters else 0}")
        logger.info(f"  Vector parameters: {list(vector_params.keys()) if vector_params else 'none'}")

        try:
            results = await self.kg_engine.execute_cypher(
                query=query,
                parameters=parameters,
                limit=limit,
                vector_params=vector_params,
            )
            logger.info(f"[OK] Query successful, returned {len(results)} records")
            return results

        except Exception as e:
            logger.error(f"[ERROR] Custom Cypher query failed: {e}")
            raise

    async def get_entity_chunks(
        self,
        entity_id: str,
        include_content: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        获取实体关联的所有 chunk 列表

        Args:
            entity_id: 实体的精确 ID（包含随机后缀，如：线段树<QyCKb7>）
            include_content: 是否包含 chunk 的完整内容（默认 False，只返回基本信息）

        Returns:
            关联的 chunk 列表，每个元素包含 chunk_id、content、tokens、file_path 等信息
        """
        if not self.initialized:
            await self.initialize()

        logger.info(f"[INFO] Getting chunks for entity: '{entity_id}' (include_content={include_content})")

        try:
            # 查询实体的 source_id 属性（包含关联的 chunk IDs）
            cypher = """
            MATCH (e:Entity {entity_id: $entity_id})
            RETURN e.source_id as source_id
            LIMIT 1
            """

            results = await self.kg_engine.execute_cypher(
                query=cypher,
                parameters={"entity_id": entity_id},
                limit=1
            )

            if not results:
                logger.warning(f"Entity '{entity_id}' not found")
                return []

            source_id_str = results[0]["source_id"]
            if not source_id_str:
                logger.info(f"Entity '{entity_id}' has no associated chunks")
                return []

            # 解析 chunk IDs - 使用多种可能的分隔方式

            chunk_ids = [chunk_id.strip() for chunk_id in source_id_str.split("<SEP>") if chunk_id.strip()]

            # 去重
            unique_chunk_ids = list(dict.fromkeys(chunk_ids))

            logger.info(f"Found {len(unique_chunk_ids)} unique chunks for entity '{entity_id}' (parsed from {len(chunk_ids)} items)")
            if len(chunk_ids) != len(unique_chunk_ids):
                logger.debug(f"Removed {len(chunk_ids) - len(unique_chunk_ids)} duplicates")

            if not include_content:
                # 只返回 chunk ID 列表（轻量级）
                return [
                    {
                        "chunk_id": chunk_id,
                        "status": "available"
                    }
                    for chunk_id in unique_chunk_ids
                ]

            # 如果需要完整内容，优先尝试从 chunks.json 文件读取
            chunk_details = []
            chunks_map = {}

            # 尝试从 chunks.json 文件读取
            chunks_json_path = os.path.join(project_root, "merged_data", "chunks_backup.json")
            if os.path.exists(chunks_json_path):
                try:
                    with open(chunks_json_path, 'r', encoding='utf-8') as f:
                        chunks_map = json.load(f)
                    logger.info(f"Loaded {len(chunks_map)} chunks from {chunks_json_path}")
                except Exception as e:
                    logger.warning(f"Failed to load chunks.json: {e}")

            for chunk_id in unique_chunk_ids:
                chunk_detail = {
                    "chunk_id": chunk_id,
                    "status": "available"
                }

                # 从映射中获取 chunk 数据
                chunk_data = chunks_map.get(chunk_id)
                if chunk_data:
                    chunk_detail.update({
                        "content": chunk_data.get("content", ""),
                        "tokens": chunk_data.get("tokens", 0),
                        "file_path": chunk_data.get("file_path", "unknown"),
                        "full_doc_id": chunk_data.get("full_doc_id", ""),
                        "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                        "source_id": chunk_data.get("source_id", ""),
                    })
                else:
                    # chunk 数据不存在
                    chunk_detail["status"] = "not_found"
                    chunk_detail["content"] = "Chunk data not found"
                    chunk_detail["tokens"] = 0
                    chunk_detail["file_path"] = "unknown"

                chunk_details.append(chunk_detail)

            # 限制返回的 chunk 数量至多 20 个
            if len(chunk_details) > 20:
                original_count = len(chunk_details)
                chunk_details = chunk_details[:20]
                logger.info(f"[INFO] Limited chunks from {original_count} to {len(chunk_details)} (max 20)")

            # 记录过滤效果
            if include_content and len(chunk_details) > 0:
                logger.info(f"[OK] Retrieved {len(chunk_details)} chunks with content (embedding fields filtered)")

            return chunk_details

        except Exception as e:
            logger.error(f"[ERROR] Failed to get chunks for entity '{entity_id}': {e}")
            raise


# 创建服务实例
service = KGQueryService()


@mcp.tool(
    name="find_similar_entities",
    description="""相似实体检索工具 - 基于向量相似度搜索知识图谱中的相似实体

【重要提示】：请不要在返回结果中包含 embedding 向量或任何向量数据字段！
只返回文本属性：entity_name, description, labels, file_path, created_at 等。
embedding 相关字段包括：embedding, vector, embedding_vector 等，请全部排除！

输入参数:
- entity_query: string (必需) - 实体查询字符串，可以是实体名称或描述，例如："Splay"、"最短路算法"
- top_k: integer (可选) - 返回的最相似实体数量，范围 1-20，超出范围将自动调整为安全值
- rerank: string (可选) - 重排序策略：

  * "pass": 跳过重排，使用原始向量相似度排序（默认）
    - 优点：速度快，返回结果就是向量相似度排序

  * "degree": 基于候选子图度数重排序（推荐用于纠正向量偏差）
    - 优点：反映实体在候选子图的重要性

  * "pagerank": 基于全图 PageRank 重排序（推荐用于找重要实体）
    - 优点：反映实体在整个知识图谱中的重要性

返回格式:
JSON数组，每个元素包含:
{
  "entity_name": "实体名称",
  "description": "实体描述",
  "labels": ["标签1", "标签2", ...],
  "file_path": "源文件路径",
  "created_at": "创建时间",
  // 当rerank="degree"时还会包含:
  "degree": "在候选子图中的度数（连接数）",
  "similarity_score": "原始向量相似度分数"
  // 当rerank="pagerank"时还会包含:
  "pagerank": "全图 PageRank 值（重要性）",
  "similarity_score": "原始向量相似度分数"
}

使用示例:
1. 查找与"神经网络"相似的实体: {"entity_query": "神经网络", "top_k": 10}
2. 查找"Python"相关的实体: {"entity_query": "Python编程语言"}
3. 查找"数据库"相关实体: {"entity_query": "数据库管理系统"}
4. 启用度数重排纠正偏差: {"entity_query": "LCT", "top_k": 10, "rerank": "degree"}
5. 启用 PageRank 重排找重要实体: {"entity_query": "图算法", "top_k": 10, "rerank": "pagerank"}

重排策略选择建议:
- 一般查询：使用 "pass" 或 "degree"
- 寻找重要概念：使用 "pagerank"
- 纠正向量偏差：使用 "degree"
"""
)
async def find_similar_entities(
    entity_query: str,
    top_k: int = 5,
    rerank: str = "pass",
) -> List[Dict[str, Any]]:
    """查找相似的实体"""
    # 安全控制：验证 top_k 参数，防止 DoS 攻击
    SAFE_DEFAULT_TOP_K = 5
    MAX_TOP_K = 20

    if top_k is None or top_k <= 0 or top_k > MAX_TOP_K:
        top_k = SAFE_DEFAULT_TOP_K if top_k is None or top_k <= 0 else MAX_TOP_K
        logger.info(f"[SECURITY] top_k adjusted to safe value: {top_k}")

    # 直接调用，避免 asyncio.wait_for 与 FastMCP 事件循环冲突
    problems = await service.find_similar_entities(
        entity_query=entity_query,
        top_k=top_k,
        rerank=rerank,
    )
    # 返回原始数据，让FastMCP自动处理序列化
    return problems
@mcp.tool(
    name="execute_custom_cypher",
    description="""高级 Cypher 查询工具 - 支持高度自定义的灵活查询（最推荐）

【重要提示】：请不要在返回结果中包含 embedding 向量或任何向量数据字段！
如果查询实体节点，只返回文本属性：entity_name, description, labels, file_path, created_at 等。

核心用法：
1. 模糊搜索实体（推荐）- 使用向量搜索
2. 精确匹配实体 - 使用参数化查询
3. 关系查询 - 查询实体间的关系

输入参数:
- query: string (必需) - Cypher 查询语句，必须使用 $参数名 格式
- parameters: object (必需) - 查询参数字典
- vector_params: object (必需) - 向量参数映射，标记哪些参数需要转换为向量进行模糊搜索。true代表嵌入成向量，false代表精确匹配
  例: {"query_vector": true} 表示将 parameters.query_vector 的字符串转换为向量
- limit: integer (可选) - 结果数量限制，范围 1-30，超出范围将自动调整为安全值

⚠️ 安全控制:
- 系统会自动验证 limit 参数，防止 DoS 攻击
- 如果传入负数、零或 None，将使用安全默认值
- 如果超出最大值（30），将被强制限制为最大值

返回格式:
JSON数组，每行包含查询结果记录

================================================================

示例1: 技巧关联查询

查询与Splay树相关的技巧有哪些：
{
  "query": "CALL db.index.vector.queryNodes('entity_embedding_index', 20, $query_vector) YIELD node as splay_node WHERE '数据结构' IN labels(splay_node) MATCH (splay_node)-[r]-(technique:Entity) WHERE '技巧' IN labels(technique) RETURN technique.entity_id as entity_name, technique.description as description, [label IN labels(technique) WHERE NOT label = 'Entity'] as labels, technique.file_path as file_path, r.weight as weight, splay_node.entity_id as related_to ORDER BY weight DESC",
  "parameters": {"query_vector": "Splay树，一种平衡二叉查找树"},
  "vector_params": {"query_vector": true},
  "limit": 20
}

1. 过滤出"数据结构"类型的节点
2. 查找这些节点的邻居中带有"技巧"标签的节点

================================================================

示例2: 精确匹配查询实体

查找特定名称的实体：
{
  "query": "MATCH (e:Entity {entity_id: $name}) RETURN e.entity_id as entity_name, e.description as description, [label IN labels(e) WHERE NOT label = 'Entity'] as labels, e.file_path as file_path",
  "parameters": {"name": "线段树<具体ID>"},
  "vector_params": {"name": false},
  "limit": 1
}

注意：精确匹配需要完整的实体ID，包含后缀的哈希值

================================================================

示例3: 查询实体关系

查询某个实体的所有关系：
{
  "query": "MATCH (e:Entity {entity_id: $name})-[r]->(target:Entity) RETURN e.entity_id as source_entity, type(r) as relationship_type, r.description as description, r.weight as weight, target.entity_id as target_entity, target ORDER BY r.weight DESC",
  "parameters": {"name": "线段树"},
  "vector_params": {"name": false},
  "limit": 20
}

关系查询：关系类型存储为Neo4j关系的type()，关键词存储在关系属性的keywords字段中

================================================================

示例4: 题目关联查询（二级邻居搜索）

查找所有和Gem Island相似的题目：
{
  "query": "MATCH (problem:Entity) WHERE '题目' IN labels(problem) AND problem.entity_id CONTAINS $problem_name MATCH (problem)-[r1]-(neighbor:Entity) MATCH (neighbor)-[r2]-(second_neighbor:Entity) WHERE '题目' IN labels(second_neighbor) RETURN second_neighbor.entity_id as entity_name, second_neighbor.description as description, [label IN labels(second_neighbor) WHERE NOT label = 'Entity'] as labels, second_neighbor.file_path as file_path, neighbor.entity_id as first_level_neighbor, r1.weight as r1_weight, r2.weight as r2_weight, (r1.weight + r2.weight) as total_weight ORDER BY total_weight DESC",
  "parameters": {"problem_name": "Gem Island"},
  "vector_params": {"problem_name": false},
  "limit": 30
}

1. 查找这些节点的直接邻居（一级关系）
2. 查找邻居的邻居（二级关系）
3. 过滤出带有"题目"标签的二级邻居节点

================================================================

示例5: 复杂查询 - 找到同时与两个实体相邻的节点

查询既可以用线段树也可以用树状数组解决的题目（使用向量搜索）：
{
  "query": "CALL db.index.vector.queryNodes('entity_embedding_index', 20, $query_vector1) YIELD node as e1 CALL db.index.vector.queryNodes('entity_embedding_index', 20, $query_vector2) YIELD node as e2 WITH collect(DISTINCT e1) as e1_list, collect(DISTINCT e2) as e2_list MATCH (e1_node)-[r1]-(common:Entity), (e2_node)-[r2]-(common) WHERE e1_node IN e1_list AND e2_node IN e2_list RETURN common.entity_id as entity_name, common.description as description, [label IN labels(common) WHERE NOT label = 'Entity'] as labels, (r1.weight + r2.weight) as total_weight ORDER BY total_weight DESC",
  "parameters": {"query_vector1": "线段树", "query_vector2": "树状数组"},
  "vector_params": {"query_vector1": true, "query_vector2": true},
  "limit": 20
}

优化说明：
- 使用较大的top_k（如20）扩大搜索范围，避免遗漏相关节点
- 通过WITH语句收集候选节点，再进行关系查询
- 可以根据需要过滤结果中的"题目"类型节点

================================================================

示例6: 精确ID的邻域共同题目查询

已知两个实体的精确ID，查找它们1-2级邻域中共有的"题目"类型节点：
{
  "query": "MATCH path1=(e1:Entity {entity_id: $entity_id1})-[*1..2]-(common:Entity) MATCH path2=(e2:Entity {entity_id: $entity_id2})-[*1..2]-(common) WHERE '题目' IN labels(common) AND common <> e1 AND common <> e2 RETURN DISTINCT common.entity_id as entity_name, common.description as description, [label IN labels(common) WHERE NOT label = 'Entity'] as labels, common.file_path as file_path, length(path1) as dist_from_e1, length(path2) as dist_from_e2 ORDER BY (length(path1) + length(path2)) ASC",
  "parameters": {"entity_id1": "线段树<QyCKb7>", "entity_id2": "树状数组<ABC123>"},
  "vector_params": {"entity_id1": false, "entity_id2": false},
  "limit": 20
}

================================================================

图谱 Schema 详细说明：

【实体类型 - 第一维度（技术分类）】
- 数据结构：数据结构相关概念
- 算法思想：通用算法思想和策略
- 动态规划：动态规划相关内容
- 图论：图论算法和概念
- 搜索：搜索算法和策略
- 字符串：字符串处理相关
- 数学：数学知识和定理
- 计算几何：几何算法

【实体类型 - 第二维度（应用层次）】
- 概念：抽象概念和定义
- 技巧：解题技巧和Trick
- 实现：具体实现方法
- 模型：数学模型和抽象模型
- 算法：具体算法名称
- 原理：原理和理论
- 题目：具体题目和实例

【关系类型（11种标准类型）】
- IS_A：分类关系（X是Y的一种/一类/实例）
- PART_OF：组成关系（包含、构成、分解）
- BASED_ON：依赖关系（基于、依赖、前提、原理）
- APPLIES_TO：应用关系（应用、解决、处理）
- EVALUATES：评估关系（验证、测试）
- EXPLAINS：解释关系（分析、阐明）
- PRACTICED_BY：实践关系（应用于题目场景）
- COMPARES_WITH：对比关系（对比、关联、类似、替代）
- LEADS_TO：推导关系（推导、转化、导致）
- OPTIMIZES：优化关系（优化、简化、加速）
- TRANSFORMS_TO：转换关系（转化、转换、映射）

Neo4j 数据存储结构：

实体存储：
- 节点标签：'Entity'（通用标签）加上多个类型标签（如：'数据结构', '技巧'）
- 节点属性：entity_id, description, file_path, created_at
- 正确获取类型：排除'Entity'标签后，[label IN labels(node) WHERE NOT label = 'Entity'] 返回所有类型标签数组

关系存储：
- 关系类型：使用 type(r) 获取（如 "APPLIES_TO", "BASED_ON"）
- 关系属性：description, weight, keywords（多个关键词用逗号分隔）
- 关系方向：支持 OUTGOING (->) 和 INCOMING (<-)

重要：不要使用 labels()[0] 直接获取类型，因为第一个标签可能是 'Entity' 或其他标签！必须使用WHERE过滤。现在实体支持多标签，返回的是标签数组而不是单独的dim1/dim2字段。

================================================================

最佳实践：
1. 优先使用向量搜索进行模糊查询，效果更好
2. 可以先使用 find_similar_entities 工具找到一个确定节点的ID，再使用Cypher工具精确匹配节点ID
3. vector_params 中设置为 true 的参数会自动转换为向量
4. 使用模糊匹配，查询的字符串描述应该尽量准确，使用“动态规划，一种用于求解复杂问题的算法思想”比直接查询“动态规划”效果更好。
5. 查询会自动添加 LIMIT 限制防止过载
6. 获取实体类型时必须排除'Entity'标签
"""
)
async def execute_custom_cypher(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    limit: Optional[int] = None,
    vector_params: Optional[Dict[str, bool]] = None,
) -> List[Dict[str, Any]]:
    """执行自定义 Cypher 查询"""

    # 直接调用，避免 asyncio.wait_for 与 FastMCP 事件循环冲突
    # 注意：结果已经在 kg_query_engine.execute_cypher 中自动过滤了 embedding 字段
    results = await service.execute_custom_cypher(
        query=query,
        parameters=parameters,
        limit=limit,
        vector_params=vector_params,
    )

    # 返回过滤后的数据，让FastMCP自动处理序列化
    return results

@mcp.tool(
    name="get_chunks",
    description="""获取实体关联的 Chunk 列表工具

根据实体的精确 ID 获取该实体来源于的所有文档块（chunks），并自动去重和过滤。支持获取完整的 chunk 内容。

输入参数:
- entity_id: string (必需) - 实体的精确 ID，必须包含随机后缀，例如："线段树<QyCKb7>"、"动态维护<x7YLqt>"
- include_content: boolean (可选) - 是否包含 chunk 的完整内容，默认 true

返回格式:
JSON数组，每个元素包含:
{
  "chunk_id": "chunk-<hash>",  // chunk 的唯一标识符
  "status": "available"        // chunk 状态
}

如果 include_content=true，返回:
{
  "chunk_id": "chunk-<hash>",
  "content": "chunk 完整文本内容",
  "tokens": 123,
  "file_path": "源文件路径",
  "full_doc_id": "文档ID",
  "chunk_order_index": 0,
  "source_id": "源ID",
  "status": "available" | "not_found"
}

使用示例:
1. 获取实体关联的 chunk ID 列表: {"entity_id": "线段树<QyCKb7>"}
2. 获取包含内容的 chunk 列表: {"entity_id": "线段树<QyCKb7>", "include_content": true}

注意事项:
- entity_id 必须是完整的实体 ID，包含尖括号和随机后缀
- 如果不知道完整 ID，可以先使用 find_similar_entities 查找实体
- chunk 列表会自动去重和过滤非 chunk 内容
- 最多返回 20 个 chunks，超出部分会被截断
"""
)
async def get_chunks(
    entity_id: str,
    include_content: bool = True,
) -> List[Dict[str, Any]]:
    """获取实体关联的 chunks"""
    # 直接调用，避免 asyncio.wait_for 与 FastMCP 事件循环冲突
    results = await service.get_entity_chunks(
        entity_id=entity_id,
        include_content=include_content,
    )
    # 返回原始数据，让FastMCP自动处理序列化
    return results

def create_embedding_func():
    """创建嵌入函数"""
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_host = os.getenv("EMBEDDING_BINDING_HOST", "https://api.siliconflow.cn/v1")
    embedding_api_key = os.getenv("EMBEDDING_BINDING_API_KEY", "")

    logger.info(f"[INFO] Embedding configuration:")
    logger.info(f"  Dimension: {embedding_dim}")
    logger.info(f"  Model: {embedding_model}")
    logger.info(f"  Service: {embedding_host}")

    async def embedding_call(texts):
        """异步 embedding 调用"""
        return await openai_embed(
            texts,
            model=embedding_model,
            base_url=embedding_host,
            api_key=embedding_api_key,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=8192,
        func=embedding_call,
    )
    return embedding_func


if __name__ == "__main__":
    # 创建 embedding 函数
    embedding_func = create_embedding_func()

    # 预初始化服务（带 embedding 支持）
    async def init_service():
        # 初始化 Neo4j 服务
        await service.initialize(embedding_func=embedding_func)
        # # 初始化 LightRAG 管理器（用于获取 chunk 内容）
        # await lightrag_manager.initialize()

    asyncio.run(init_service())
    import asyncio
    # 从环境变量读取MCP端口配置
    mcp_port = int(os.getenv("MCP_PORT", "8000"))
    asyncio.run(mcp.run_async(
            transport="streamable-http",
            host="127.0.0.1",
            port=mcp_port,
            log_level="debug",
            stateless_http=True
        ))
