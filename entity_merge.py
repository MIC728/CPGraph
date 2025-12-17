"""
实体合并模块 - 处理新架构提取的实体和关系数据

从新架构 (multi_threaded_extractor.py) 输出的 JSON 文件加载数据，
进行实体合并、去重、类型校正等处理。
"""

import os
import asyncio
import json
import re
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
import time
from concurrent.futures import ThreadPoolExecutor

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

# LightRAG 常量和提示词（需要用于实体合并逻辑）
from lightrag.prompt import PROMPTS
from lightrag.constants import (
    DEFAULT_ENTITY_TYPES_DIM1,
    DEFAULT_ENTITY_TYPES_DIM2,
    DEFAULT_ENTITY_MERGE_TOPK,
    DEFAULT_ENTITY_MERGE_MAX_ASYNC,
)

# 导入LLM摘要服务
from llm_summary_service import LLMSummaryService

# 设置日志（非缓冲模式，解决多线程日志延迟问题）
import sys

class UnbufferedHandler(logging.StreamHandler):
    """非缓冲日志处理器，解决多线程日志输出延迟问题"""
    def emit(self, record):
        super().emit(record)
        self.flush()  # 立即刷新输出

# 配置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    handlers=[UnbufferedHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ==================== 增量更新配置 ====================

def get_incremental_config() -> Dict[str, Any]:
    """
    从环境变量获取增量更新配置

    Returns:
        Dict[str, Any]: 配置字典，包含:
            - incremental: bool, 是否启用增量更新
            - merged_data_dir: str, merged_data 目录路径
    """
    incremental_str = os.environ.get("ENTITY_MERGE_INCREMENTAL", "false").lower()
    incremental = incremental_str in ("true", "1", "yes")

    merged_data_dir = os.environ.get("MERGED_DATA_DIR", "./merged_data")

    logger.info(f"增量更新配置: incremental={incremental}, merged_data_dir={merged_data_dir}")

    return {
        "incremental": incremental,
        "merged_data_dir": merged_data_dir
    }


def load_existing_merged_data(merged_data_dir: str) -> Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
    """
    从 merged_data 目录加载已有的实体、关系和chunks数据

    Args:
        merged_data_dir: merged_data 目录路径

    Returns:
        Tuple[List[Dict], List[Dict], Dict[str, Dict]]:
            (entities, relations, chunks)
            - entities: 实体列表（embedding已解码为list）
            - relations: 关系列表
            - chunks: chunk字典 {chunk_id: chunk_data}
    """
    entities = []
    relations = []
    chunks = {}

    # 确保目录存在
    if not os.path.exists(merged_data_dir):
        logger.info(f"merged_data 目录不存在: {merged_data_dir}，将创建新目录")
        os.makedirs(merged_data_dir, exist_ok=True)
        return entities, relations, chunks

    # 加载实体
    entities_file = os.path.join(merged_data_dir, "processed_entities.json")
    if os.path.exists(entities_file):
        try:
            with open(entities_file, 'r', encoding='utf-8') as f:
                entities = json.load(f)

            # 解码 base64 embedding
            for entity in entities:
                if "embedding" in entity and isinstance(entity["embedding"], str):
                    entity["embedding"] = embed_base64_to_list(entity["embedding"])

            logger.info(f"从 {entities_file} 加载 {len(entities)} 个已有实体")
        except Exception as e:
            logger.warning(f"加载已有实体失败: {e}")
            entities = []

    # 加载关系
    relations_file = os.path.join(merged_data_dir, "processed_relations.json")
    if os.path.exists(relations_file):
        try:
            with open(relations_file, 'r', encoding='utf-8') as f:
                relations = json.load(f)
            logger.info(f"从 {relations_file} 加载 {len(relations)} 个已有关系")
        except Exception as e:
            logger.warning(f"加载已有关系失败: {e}")
            relations = []

    # 加载 chunks 备份
    chunks_file = os.path.join(merged_data_dir, "chunks_backup.json")
    if os.path.exists(chunks_file):
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
            logger.info(f"从 {chunks_file} 加载 {len(chunks)} 个已有chunks")
        except Exception as e:
            logger.warning(f"加载已有chunks失败: {e}")
            chunks = {}

    return entities, relations, chunks


def merge_with_existing_data(
    new_entities: List[Dict],
    new_relations: List[Dict],
    existing_entities: List[Dict],
    existing_relations: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """
    将新数据与已有数据合并（去重）

    去重策略：
    - entities: 按 entity_id 去重，新数据优先（更新场景）
    - relations: 按 (src_id, tgt_id) 去重，新数据优先

    Args:
        new_entities: 新提取的实体列表
        new_relations: 新提取的关系列表
        existing_entities: 已有实体列表
        existing_relations: 已有关系列表

    Returns:
        Tuple[List[Dict], List[Dict]]: (merged_entities, merged_relations)
    """
    # 合并实体：按 entity_id 去重
    entity_map = {}

    # 先加载已有实体
    for entity in existing_entities:
        entity_id = entity.get("entity_id") or entity.get("id")
        if entity_id:
            entity_map[entity_id] = entity

    existing_count = len(entity_map)

    # 新实体覆盖已有实体（更新场景）
    new_count = 0
    update_count = 0
    for entity in new_entities:
        entity_id = entity.get("entity_id") or entity.get("id")
        if entity_id:
            if entity_id in entity_map:
                update_count += 1
            else:
                new_count += 1
            entity_map[entity_id] = entity

    merged_entities = list(entity_map.values())

    logger.info(f"实体合并: 已有 {existing_count}, 新增 {new_count}, 更新 {update_count}, 合并后 {len(merged_entities)}")

    # 合并关系：按 (src_id, tgt_id) 去重
    relation_map = {}

    # 先加载已有关系
    for relation in existing_relations:
        src_id = relation.get("src_id")
        tgt_id = relation.get("tgt_id")
        if src_id and tgt_id:
            key = (src_id, tgt_id)
            relation_map[key] = relation

    existing_rel_count = len(relation_map)

    # 新关系覆盖已有关系
    new_rel_count = 0
    update_rel_count = 0
    for relation in new_relations:
        src_id = relation.get("src_id")
        tgt_id = relation.get("tgt_id")
        if src_id and tgt_id:
            key = (src_id, tgt_id)
            if key in relation_map:
                update_rel_count += 1
            else:
                new_rel_count += 1
            relation_map[key] = relation

    merged_relations = list(relation_map.values())

    logger.info(f"关系合并: 已有 {existing_rel_count}, 新增 {new_rel_count}, 更新 {update_rel_count}, 合并后 {len(merged_relations)}")

    return merged_entities, merged_relations


def merge_chunks(
    new_chunks: Dict[str, Dict],
    existing_chunks: Dict[str, Dict]
) -> Dict[str, Dict]:
    """
    合并 chunks 数据（保留全部，按 chunk_id 去重）

    Args:
        new_chunks: 新的 chunks 字典
        existing_chunks: 已有的 chunks 字典

    Returns:
        Dict[str, Dict]: 合并后的 chunks 字典
    """
    merged = existing_chunks.copy()

    new_count = 0
    for chunk_id, chunk_data in new_chunks.items():
        if chunk_id not in merged:
            new_count += 1
        merged[chunk_id] = chunk_data

    logger.info(f"Chunks合并: 已有 {len(existing_chunks)}, 新增 {new_count}, 合并后 {len(merged)}")

    return merged


class EntityDataLoader:
    """
    实体数据加载器 - 从新架构输出加载实体和关系数据

    支持两种加载方式：
    1. 从 JSON 文件加载（新架构输出的 entities.json, relations.json）
    2. 直接从字典加载（extract_documents_async() 的输出）
    """

    def __init__(self, output_dir: str = None):
        """
        初始化数据加载器

        Args:
            output_dir: 输出目录，默认从环境变量读取
        """
        # 默认从环境变量读取，如果没有则使用 "./processed_data"
        self.output_dir = output_dir or os.getenv("ENTITY_MERGE_PROCESSED_DATA_DIR", "./processed_data")

        # 数据存储
        self.entities: List[Dict[str, Any]] = []
        self.relations: List[Dict[str, Any]] = []
        self.chunks: List[Dict[str, Any]] = []  # 存储chunks数据

        # 获取增量更新配置
        self.incremental_config = get_incremental_config()
        self.incremental_enabled = self.incremental_config["incremental"]
        self.merged_data_dir = self.incremental_config["merged_data_dir"]

        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        logger.info(f"初始化实体数据加载器，输出目录: {self.output_dir}")
        logger.info(f"增量更新配置: enabled={self.incremental_enabled}, data_dir={self.merged_data_dir}")

    def load_from_json(
        self,
        entities_file: str,
        relations_file: str = None,
        chunks_file: str = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        从新架构生成的 JSON 文件加载实体和关系数据

        Args:
            entities_file: 实体 JSON 文件路径，如 "./new_rag_storage/entities.json"
            relations_file: 关系 JSON 文件路径，如 "./new_rag_storage/relations.json"
            chunks_file: chunks JSON 文件路径，如 "./new_rag_storage/chunks.json"

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (entities, relations)
        """
        try:
            # 加载实体数据
            if entities_file and os.path.exists(entities_file):
                with open(entities_file, 'r', encoding='utf-8') as f:
                    new_entities = json.load(f)
                logger.info(f"从 JSON 文件加载 {len(new_entities)} 个新实体: {entities_file}")
            else:
                raise FileNotFoundError(f"实体文件不存在: {entities_file}")

            # 加载关系数据
            new_relations = []
            if relations_file and os.path.exists(relations_file):
                with open(relations_file, 'r', encoding='utf-8') as f:
                    new_relations = json.load(f)
                logger.info(f"从 JSON 文件加载 {len(new_relations)} 个新关系: {relations_file}")
            else:
                logger.warning(f"关系文件不存在或未指定: {relations_file}")

            # 加载chunks数据
            new_chunks = []
            if chunks_file and os.path.exists(chunks_file):
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    new_chunks = json.load(f)
                logger.info(f"从 JSON 文件加载 {len(new_chunks)} 个新chunks: {chunks_file}")
            else:
                # 自动推断chunks文件路径（与实体文件在同一目录）
                if entities_file:
                    chunks_file = os.path.join(os.path.dirname(entities_file), "chunks.json")
                    if os.path.exists(chunks_file):
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            new_chunks = json.load(f)
                        logger.info(f"从 JSON 文件加载 {len(new_chunks)} 个新chunks: {chunks_file}")
                if not new_chunks:
                    logger.info("未找到chunks文件，将不处理chunks数据")

            # 检查是否启用增量更新
            if self.incremental_enabled:
                logger.info("增量更新模式：加载已有数据并合并")

                # 加载已有数据
                existing_entities, existing_relations, existing_chunks = load_existing_merged_data(self.merged_data_dir)

                # 合并数据
                self.entities, self.relations = merge_with_existing_data(
                    new_entities=new_entities,
                    new_relations=new_relations,
                    existing_entities=existing_entities,
                    existing_relations=existing_relations
                )

                # 存储chunks数据
                if new_chunks or existing_chunks:
                    # 确保new_chunks是字典格式
                    if isinstance(new_chunks, list):
                        new_chunks_dict = {chunk.get("chunk_id", idx): chunk for idx, chunk in enumerate(new_chunks)}
                    else:
                        new_chunks_dict = new_chunks

                    # 合并chunks数据
                    all_chunks = new_chunks_dict
                    if existing_chunks:
                        # 合并字典格式的chunks
                        if isinstance(existing_chunks, dict):
                            # 以字典形式合并，新chunk覆盖旧chunk
                            all_chunks = {**existing_chunks, **new_chunks_dict}
                        elif isinstance(existing_chunks, list):
                            # 如果是列表，转为字典（兼容旧版本）
                            existing_dict = {chunk.get("chunk_id", idx): chunk for idx, chunk in enumerate(existing_chunks)}
                            all_chunks = {**existing_dict, **new_chunks_dict}
                    self.chunks = all_chunks
                    logger.info(f"已合并chunks数据: {len(self.chunks)} 个chunks")

            else:
                # 非增量更新模式，直接使用新数据
                self.entities = new_entities
                self.relations = new_relations
                self.chunks = new_chunks
                logger.info(f"非增量更新模式：使用新数据 - {len(new_entities)} 实体, {len(new_relations)} 关系, {len(new_chunks)} chunks")

            # 标准化数据格式
            self.entities = self._standardize_entities(self.entities)
            self.relations = self._standardize_relations(self.relations)

            logger.info(f"数据加载完成: {len(self.entities)} 实体, {len(self.relations)} 关系, {len(self.chunks)} chunks")
            return self.entities, self.relations

        except Exception as e:
            logger.error(f"从 JSON 文件加载数据失败: {e}")
            raise

    def load_from_dict(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        直接从字典列表加载实体和关系数据

        可以直接接收 extract_documents_async() 的输出。

        Args:
            entities: 实体列表
            relations: 关系列表

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (entities, relations)
        """
        try:
            self.entities = entities if entities else []
            self.relations = relations if relations else []

            # 标准化数据格式
            self.entities = self._standardize_entities(self.entities)
            self.relations = self._standardize_relations(self.relations)

            logger.info(f"从字典加载数据完成: {len(self.entities)} 实体, {len(self.relations)} 关系")
            return self.entities, self.relations

        except Exception as e:
            logger.error(f"从字典加载数据失败: {e}")
            raise

    async def generate_embeddings(
        self,
        embedding_func: Callable,
        batch_size: int = 32,
        max_concurrent: int = 20
    ) -> int:
        """
        为所有实体批量异步生成嵌入向量

        Args:
            embedding_func: 嵌入函数，接收文本列表，返回向量列表
            batch_size: 每批处理的实体数量
            max_concurrent: 最大并发批次数

        Returns:
            int: 成功生成嵌入的实体数量
        """
        if not self.entities:
            logger.warning("没有实体需要生成嵌入向量")
            return 0

        # 过滤出没有嵌入向量的实体
        entities_without_embedding = [
            e for e in self.entities if e.get("embedding") is None
        ]

        if not entities_without_embedding:
            logger.info("所有实体已有嵌入向量")
            return len(self.entities)

        logger.info(f"开始为 {len(entities_without_embedding)} 个实体生成嵌入向量...")
        logger.info(f"配置: batch_size={batch_size}, max_concurrent={max_concurrent}")

        # 准备文本内容
        texts_to_embed = []
        entity_indices = []  # 记录对应的实体在 self.entities 中的索引

        entity_id_to_idx = {
            e.get("entity_id") or e.get("id"): i
            for i, e in enumerate(self.entities)
        }

        for entity in entities_without_embedding:
            entity_id = entity.get("entity_id") or entity.get("id") or ""
            description = entity.get("description", "")
            # 构造嵌入文本：实体名 + 描述
            text = f"{entity_id}\n{description}".strip()
            if text:
                texts_to_embed.append(text)
                entity_indices.append(entity_id_to_idx.get(entity_id))

        if not texts_to_embed:
            logger.warning("没有有效的文本内容用于生成嵌入")
            return 0

        # 分批处理
        total_batches = (len(texts_to_embed) + batch_size - 1) // batch_size
        logger.info(f"总计 {len(texts_to_embed)} 个文本，分 {total_batches} 批处理")

        semaphore = asyncio.Semaphore(max_concurrent)
        successful_count = 0
        failed_count = 0

        async def process_batch(batch_idx: int, batch_texts: List[str], batch_indices: List[int]):
            """处理单个批次"""
            nonlocal successful_count, failed_count

            async with semaphore:
                try:
                    # 调用嵌入函数
                    if asyncio.iscoroutinefunction(embedding_func):
                        embeddings = await embedding_func(batch_texts)
                    else:
                        # 如果是同步函数，在线程池中执行
                        loop = asyncio.get_event_loop()
                        embeddings = await loop.run_in_executor(None, embedding_func, batch_texts)

                    # 如果返回的是协程，await 它
                    if asyncio.iscoroutine(embeddings):
                        embeddings = await embeddings

                    # 将嵌入向量写回实体
                    if embeddings is not None and len(embeddings) == len(batch_texts):
                        for i, idx in enumerate(batch_indices):
                            if idx is not None:
                                # 处理可能是 numpy array 或 list 的情况
                                embedding = embeddings[i]
                                if hasattr(embedding, 'tolist'):
                                    embedding = embedding.tolist()
                                self.entities[idx]["embedding"] = embedding
                                successful_count += 1
                            else:
                                failed_count += 1
                        logger.debug(f"批次 {batch_idx + 1}/{total_batches} 完成: {len(batch_texts)} 个")
                    else:
                        failed_count += len(batch_texts)
                        logger.warning(f"批次 {batch_idx + 1} 返回的嵌入数量不匹配")

                except Exception as e:
                    failed_count += len(batch_texts)
                    logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")

        # 创建所有批次的任务
        tasks = []
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(texts_to_embed))
            batch_texts = texts_to_embed[start:end]
            batch_indices = entity_indices[start:end]
            tasks.append(process_batch(batch_idx, batch_texts, batch_indices))

        # 并发执行所有批次
        start_time = time.time()
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time

        logger.info(f"嵌入生成完成: 成功 {successful_count}, 失败 {failed_count}")
        logger.info(f"耗时: {elapsed:.2f}秒, 速度: {successful_count / elapsed:.1f} 实体/秒")

        return successful_count

    def _standardize_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标准化实体数据格式 - 保持原始格式简洁性
        
        Args:
            entities: 原始实体列表
            
        Returns:
            List[Dict[str, Any]]: 标准化后的实体列表
        """
        standardized = []
        
        for entity in entities:
            try:
                # 确保必要的字段存在
                entity_id = entity.get("entity_id") or entity.get("id") or entity.get("entity_name", "")
                
                if not entity_id:
                    logger.warning(f"跳过无ID的实体: {entity}")
                    continue
                
                # 保持原始格式，只保留核心字段
                standardized_entity = {
                    "entity_id": entity_id,
                    "id": entity.get("id", entity_id),
                    "entity_type_dim1": entity.get("entity_type_dim1", ""),
                    "entity_type_dim2": entity.get("entity_type_dim2", ""),
                    "description": entity.get("description", ""),
                    "source_id": entity.get("source_id", ""),
                    "file_path": entity.get("file_path", ""),
                    "created_at": entity.get("created_at", ""),
                    "truncate": entity.get("truncate", ""),
                    "is_problem_extracted": entity.get("is_problem_extracted", False)  # 保留题目提取标记
                }

                # 只添加非空的核心字段（但保留 is_problem_extracted 即使为 False）
                standardized_entity = {k: v for k, v in standardized_entity.items()
                                       if v != "" or k == "is_problem_extracted"}
                
                standardized.append(standardized_entity)
                
            except Exception as e:
                logger.warning(f"标准化实体失败 {entity}: {e}")
                continue
        
        return standardized

    def _standardize_relations(self, relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        标准化关系数据格式
        
        Args:
            relations: 原始关系列表
            
        Returns:
            List[Dict[str, Any]]: 标准化后的关系列表
        """
        standardized = []
        
        for relation in relations:
            try:
                # 确保必要的字段存在 - 修复字段名映射
                src_id = (relation.get("src_id") or 
                         relation.get("source") or      # 来自知识图谱的source字段
                         relation.get("source_id") or 
                         relation.get("src", ""))
                
                tgt_id = (relation.get("tgt_id") or 
                         relation.get("target") or      # 来自知识图谱的target字段
                         relation.get("target_id") or 
                         relation.get("tgt", ""))
                
                # 如果仍然没有找到ID，跳过该关系
                if not src_id or not tgt_id:
                    logger.warning(f"跳过无源或目标ID的关系: {relation}")
                    continue
                
                # 标准化字段名 - 只保留必要字段，删除兼容字段
                standardized_relation = {
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "description": relation.get("description", ""),
                    "keywords": relation.get("keywords", ""),
                    "weight": float(relation.get("weight", 1.0)),
                    "source_chunk_id": relation.get("source_id", ""),  # 源文档块ID
                    "file_path": relation.get("file_path", ""),
                    "content": relation.get("content", ""),
                    "data_source": relation.get("data_source", "kg"),

                    # 保留所有原始字段
                    **{k: v for k, v in relation.items() if k not in [
                        "src_id", "tgt_id", "source", "target", "source_id", "target_id", "src", "tgt", "description",
                        "keywords", "weight", "source_chunk_id", "file_path", "content", "data_source",
                    ]}
                }
                
                # 跳过自我关系
                if standardized_relation["src_id"] == standardized_relation["tgt_id"]:
                    logger.debug(f"跳过自我关系: {src_id} -> {tgt_id}")
                    continue
                
                standardized.append(standardized_relation)
                
            except Exception as e:
                logger.warning(f"标准化关系失败 {relation}: {e}")
                continue
        
        return standardized

    def save_data(self, entities_file: str = None, relations_file: str = None, chunks: list = None):
        """
        保存数据到文件

        Args:
            entities_file: 实体文件路径，默认使用 output_dir/processed_entities.json
            relations_file: 关系文件路径，默认使用 output_dir/processed_relations.json
            chunks: 可选的chunks数据列表，如果提供则保存到chunks_backup.json
        """
        entities_file = entities_file or os.path.join(self.output_dir, "processed_entities.json")
        relations_file = relations_file or os.path.join(self.output_dir, "processed_relations.json")

        try:
            # 转换embedding为base64格式以减少存储空间
            entities_to_save = []
            for entity in self.entities:
                entity_copy = entity.copy()
                if "embedding" in entity_copy and isinstance(entity_copy["embedding"], list):
                    # 将embedding列表转为base64字符串
                    entity_copy["embedding"] = embed_list_to_base64(entity_copy["embedding"])
                entities_to_save.append(entity_copy)

            with open(entities_file, 'w', encoding='utf-8') as f:
                json.dump(entities_to_save, f, ensure_ascii=False, indent=2)
            logger.info(f"实体数据已保存到: {entities_file} ({len(self.entities)} 条记录，embedding已压缩为base64)")

            with open(relations_file, 'w', encoding='utf-8') as f:
                json.dump(self.relations, f, ensure_ascii=False, indent=2)
            logger.info(f"关系数据已保存到: {relations_file} ({len(self.relations)} 条记录)")

            # 始终保存chunks备份（不管是否启用增量更新）
            # 如果没有传入chunks参数，使用self.chunks
            chunks_to_save = chunks if chunks is not None else self.chunks

            if chunks_to_save:
                # chunks数据是字典格式：{chunk_id: chunk_obj, ...}
                # 直接保存，无需处理embedding
                chunks_backup_file = os.path.join(self.output_dir, "chunks_backup.json")
                with open(chunks_backup_file, 'w', encoding='utf-8') as f:
                    json.dump(chunks_to_save, f, ensure_ascii=False, indent=2)
                logger.info(f"chunks备份已保存到: {chunks_backup_file} ({len(chunks_to_save)} 个chunk)")
            else:
                logger.info("没有chunks数据需要备份")

        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            raise

    def print_summary(self):
        """打印数据摘要"""
        print("\n" + "="*60)
        print("实体数据加载摘要")
        print("="*60)
        print(f"输出目录: {self.output_dir}")
        print(f"实体数量: {len(self.entities)}")
        print(f"关系数量: {len(self.relations)}")

        if self.entities:
            print("\n实体类型分布:")
            type_counts = {}
            for entity in self.entities:
                type1 = entity.get("entity_type_dim1", "UNKNOWN")
                type2 = entity.get("entity_type_dim2", "UNKNOWN")
                type_key = f"{type1}/{type2}"
                type_counts[type_key] = type_counts.get(type_key, 0) + 1

            for type_key, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {type_key}: {count}")

        if self.relations:
            print(f"\n关系权重分布:")
            weight_ranges = {"0-1": 0, "1-2": 0, "2-3": 0, "3+": 0}
            for relation in self.relations:
                weight = relation.get("weight", 1.0)
                if weight < 1:
                    weight_ranges["0-1"] += 1
                elif weight < 2:
                    weight_ranges["1-2"] += 1
                elif weight < 3:
                    weight_ranges["2-3"] += 1
                else:
                    weight_ranges["3+"] += 1

            for range_name, count in weight_ranges.items():
                print(f"  {range_name}: {count}")

        print("="*60)


# 保留旧类名作为别名，方便兼容
LightRAGDataExtractor = EntityDataLoader


# ==================== 双策略实体合并配置 ====================

class DualStrategyMergeConfig:
    """双策略实体合并配置"""

    # 题目实体合并阈值
    PROBLEM_MERGE_THRESHOLD = 0.8

    # TopK设置
    NORMAL_ENTITY_TOPK = 3      # 非题目实体：检索3个候选
    PROBLEM_ENTITY_TOPK = 1     # 题目实体：只检索1个最相似

    # 小合并组阈值
    SMALL_MERGE_THRESHOLD = 4


def chunk_entities_by_problem_flag(entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按问题提取标记将实体数据分块

    Args:
        entities: 实体列表

    Returns:
        Dict[str, List[Dict[str, Any]]]: 按问题提取标记分块的字典
    """
    normal_entities = []
    problem_entities = []

    for entity in entities:
        if entity.get("is_problem_extracted", False):
            problem_entities.append(entity)
        else:
            normal_entities.append(entity)

    return {
        "normal_entities": normal_entities,  # 先处理
        "problem_entities": problem_entities  # 后处理
    }


def process_entity_chunk_with_dual_strategy(
    chunk_name: str,
    entities: List[Dict[str, Any]],
    config: DualStrategyMergeConfig = None
) -> Dict[str, Any]:
    """
    使用双策略处理单个实体分块

    Args:
        chunk_name: 分块名称
        entities: 该分块中的实体列表
        config: 双策略合并配置

    Returns:
        Dict[str, Any]: 处理结果
    """
    import hnswlib

    config = config or DualStrategyMergeConfig()

    # 分离题目实体和非题目实体
    normal_entities = [e for e in entities if not e.get("is_problem_extracted", False)]
    problem_entities = [e for e in entities if e.get("is_problem_extracted", False)]

    logger.info(f"分块 '{chunk_name}' 双策略处理:")
    logger.info(f"  - 普通实体: {len(normal_entities)}")
    logger.info(f"  - 题目实体: {len(problem_entities)}")

    # 处理普通实体（高质量策略）
    normal_result = None
    if normal_entities:
        normal_result = process_normal_entities_chunk(chunk_name, normal_entities, config)

    # 处理题目实体（快速策略）
    problem_result = None
    if problem_entities:
        problem_result = process_problem_entities_chunk(chunk_name, problem_entities, config)

    # 合并结果
    merged_entities = []
    if normal_result:
        merged_entities.extend(normal_result.get("merged_entities", []))
    if problem_result:
        merged_entities.extend(problem_result.get("merged_entities", []))

    return {
        "chunk_name": chunk_name,
        "entity_count": len(entities),
        "normal_entity_count": len(normal_entities),
        "problem_entity_count": len(problem_entities),
        "status": "processed",
        "merged_entities": merged_entities,
        "normal_result": normal_result,
        "problem_result": problem_result
    }


def process_normal_entities_chunk(
    chunk_name: str,
    entities: List[Dict[str, Any]],
    config: DualStrategyMergeConfig
) -> Dict[str, Any]:
    """
    处理普通实体（高质量策略 - 全部使用LLM）

    Args:
        chunk_name: 分块名称
        entities: 普通实体列表
        config: 双策略合并配置

    Returns:
        Dict[str, Any]: 处理结果
    """
    import hnswlib

    # 创建向量检索实例
    entities_with_embeddings = [e for e in entities if e.get("embedding") is not None]

    if not entities_with_embeddings:
        logger.warning(f"分块 '{chunk_name}' 没有包含嵌入向量的普通实体")
        return {
            "chunk_name": chunk_name,
            "entity_count": len(entities),
            "status": "skipped",
            "merged_entities": entities
        }

    # 初始化HNSW索引
    embedding_dim = len(entities_with_embeddings[0]["embedding"])
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.init_index(max_elements=len(entities_with_embeddings), ef_construction=200, M=16)
    index.set_ef(50)

    # 并查集和数据结构
    entities_map = {}
    entities_dict = {}

    for entity in entities:
        entity_id = entity.get('entity_id')
        entities_map[entity_id] = entity_id
        entities_dict[entity_id] = entity

    logger.info(f"普通实体分块 '{chunk_name}' 使用高质量策略，包含 {len(entities_with_embeddings)} 个实体")

    # 普通实体策略：只检索1个最相似实体，直接调用LLM评估
    for idx, entity in enumerate(entities_with_embeddings):
        entity_id = entity.get("entity_id")

        # 查询相似实体（只查询已插入的节点）
        similar_entities = []
        if entity.get("embedding") is not None:
            if idx > 0:  # 有已插入的实体时才查询
                try:
                    candidates, distances = index.knn_query(entity["embedding"], k=config.NORMAL_ENTITY_TOPK)
                    similar_entities = [entities_with_embeddings[int(c)] for c in candidates[0]
                                      if int(c) < len(entities_with_embeddings) and int(c) < idx]

                    logger.debug(f"  普通实体 {entity_id}: 找到 {len(similar_entities)} 个相似实体")
                except Exception as e:
                    logger.warning(f"普通实体 {entity_id} HNSW查询失败: {e}")

        # 普通实体策略：直接映射到自身（不合并，等待LLM阶段统一处理）
        # 这里只建立映射关系，合并逻辑在LLM阶段统一处理
        if similar_entities:
            # 记录相似关系，但不做合并决策
            best_match_entity = similar_entities[0]
            best_match_entity_id = best_match_entity.get("entity_id")
            entities_map[entity_id] = best_match_entity_id

        # 插入当前实体到HNSW
        if entity.get("embedding") is not None:
            index.add_items(entity["embedding"], idx)

    # 构建合并组
    merge_groups = {}
    for entity_id, root_id in entities_map.items():
        if root_id not in merge_groups:
            merge_groups[root_id] = []
        merge_groups[root_id].append(entity_id)

    # 实际执行实体合并
    merged_entities = []
    for root_id, group in merge_groups.items():
        if len(group) == 1:
            # 单个实体直接保留
            merged_entities.append(entities_dict[group[0]])
        else:
            # 多个实体需要合并
            merged_entity = merge_entity_group(group, entities_dict, f"{chunk_name}_normal")
            merged_entities.append(merged_entity)
            logger.info(f"  → 普通实体合并: {merged_entity.get('entity_id', 'unknown')} (包含{len(group)}个实体)")

    logger.info(f"普通实体分块 '{chunk_name}' 处理完成，最终实体数: {len(merged_entities)}")

    return {
        "chunk_name": chunk_name,
        "entity_count": len(entities),
        "status": "processed",
        "merged_entities": merged_entities,
        "strategy": "high_quality_llm"
    }


def process_problem_entities_chunk(
    chunk_name: str,
    entities: List[Dict[str, Any]],
    config: DualStrategyMergeConfig
) -> Dict[str, Any]:
    """
    处理题目实体（快速策略 - 只基于相似度阈值，完全不使用LLM）

    Args:
        chunk_name: 分块名称
        entities: 题目实体列表
        config: 双策略合并配置

    Returns:
        Dict[str, Any]: 处理结果
    """
    import hnswlib

    # 创建向量检索实例
    entities_with_embeddings = [e for e in entities if e.get("embedding") is not None]

    if not entities_with_embeddings:
        logger.warning(f"分块 '{chunk_name}' 没有包含嵌入向量的题目实体")
        return {
            "chunk_name": chunk_name,
            "entity_count": len(entities),
            "status": "skipped",
            "merged_entities": entities
        }

    # 初始化HNSW索引
    embedding_dim = len(entities_with_embeddings[0]["embedding"])
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.init_index(max_elements=len(entities_with_embeddings), ef_construction=200, M=16)
    index.set_ef(50)

    # 并查集和数据结构
    entities_map = {}
    entities_dict = {}

    for entity in entities:
        entity_id = entity.get('entity_id')
        entities_map[entity_id] = entity_id
        entities_dict[entity_id] = entity

    logger.info(f"题目实体分块 '{chunk_name}' 使用快速策略，包含 {len(entities_with_embeddings)} 个实体")

    # 题目实体策略：只检索1个最相似实体，相似度>阈值才合并
    auto_merged_count = 0
    auto_skipped_count = 0

    for idx, entity in enumerate(entities_with_embeddings):
        entity_id = entity.get("entity_id")

        # 查询最相似实体（只查询已插入的节点）
        if entity.get("embedding") is not None:
            if idx > 0:  # 有已插入的实体时才查询
                try:
                    candidates, distances = index.knn_query(entity["embedding"], k=config.PROBLEM_ENTITY_TOPK)
                    if len(candidates[0]) > 0:
                        # 获取最相似的实体
                        best_match_idx = candidates[0][0]
                        best_distance = distances[0][0]

                        # 将距离转换为相似度
                        similarity = 1.0 - best_distance

                        # 快速决策：只基于相似度阈值
                        if similarity > config.PROBLEM_MERGE_THRESHOLD:
                            best_match_entity = entities[best_match_idx]
                            best_match_entity_id = best_match_entity.get("entity_id")

                            if best_match_entity_id:
                                entities_map[entity_id] = best_match_entity_id
                                auto_merged_count += 1
                                logger.debug(f"  ✓ 快速合并: {entity_id} -> {best_match_entity_id} (相似度: {similarity:.3f})")
                        else:
                            auto_skipped_count += 1
                            logger.debug(f"  ○ 快速跳过: {entity_id} (相似度: {similarity:.3f})")

                except Exception as e:
                    logger.warning(f"题目实体 {entity_id} HNSW查询失败: {e}")

        # 插入当前实体到HNSW
        if entity.get("embedding") is not None:
            index.add_items(entity["embedding"], idx)

    # 构建合并组
    merge_groups = {}
    for entity_id, root_id in entities_map.items():
        if root_id not in merge_groups:
            merge_groups[root_id] = []
        merge_groups[root_id].append(entity_id)

    # 实际执行实体合并（不使用LLM，直接拼接）
    merged_entities = []
    for root_id, group in merge_groups.items():
        if len(group) == 1:
            # 单个实体直接保留
            merged_entities.append(entities_dict[group[0]])
        else:
            # 多个实体直接拼接（不使用LLM）
            group_entities = [entities_dict[eid] for eid in group if eid in entities_dict]

            # 选择主实体
            main_entity = group_entities[0].copy()

            # 合并描述（直接拼接）
            descriptions = [e.get('description', '') for e in group_entities if e.get('description')]
            if descriptions:
                main_entity['description'] = "<SEP>".join(descriptions)

            # 合并类型（去重）
            all_dim1_types = set()
            all_dim2_types = set()
            for entity in group_entities:
                dim1 = entity.get('entity_type_dim1', '')
                dim2 = entity.get('entity_type_dim2', '')
                if dim1:
                    all_dim1_types.update([t.strip() for t in dim1.replace('，', ',').split(',') if t.strip()])
                if dim2:
                    all_dim2_types.update([t.strip() for t in dim2.replace('，', ',').split(',') if t.strip()])

            main_entity['entity_type_dim1'] = ','.join(sorted(all_dim1_types))
            main_entity['entity_type_dim2'] = ','.join(sorted(all_dim2_types))

            # 记录合并信息
            main_entity['merged_from'] = group
            main_entity['merge_count'] = len(group)
            main_entity['merge_timestamp'] = int(time.time())
            main_entity['merge_method'] = 'fast_concat'  # 标记为快速拼接合并

            merged_entities.append(main_entity)
            logger.debug(f"  ✓ 快速合并: {group} -> {main_entity.get('entity_id', 'unknown')} ({len(group)}个实体)")

    logger.info(f"题目实体分块 '{chunk_name}' 快速处理完成:")
    logger.info(f"  - 自动合并: {auto_merged_count}")
    logger.info(f"  - 自动跳过: {auto_skipped_count}")
    logger.info(f"  - 最终实体数: {len(merged_entities)}")

    return {
        "chunk_name": chunk_name,
        "entity_count": len(entities),
        "status": "processed",
        "merged_entities": merged_entities,
        "strategy": "fast_similarity",
        "auto_merged_count": auto_merged_count,
        "auto_skipped_count": auto_skipped_count
    }


async def process_entity_chunks_with_dual_strategy(
    entity_chunks: Dict[str, List[Dict[str, Any]]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None,
    topk: int = None,
    max_concurrent: int = None,
    max_workers: int = None,
    dual_config: DualStrategyMergeConfig = None
) -> Dict[str, Dict[str, Any]]:
    """
    使用双策略多线程处理所有实体分块

    普通实体使用LLM高质量策略，题目实体使用快速相似度策略

    Args:
        entity_chunks: 按dim1类型分块的实体字典
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表
        topk: 检索相似实体的数量
        max_concurrent: 分块内LLM最大并发数
        max_workers: 分块间最大线程数
        dual_config: 双策略配置

    Returns:
        Dict[str, Dict[str, Any]]: 每个分块的处理结果
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    dual_config = dual_config or DualStrategyMergeConfig()

    if max_workers is None:
        max_workers = min(len(entity_chunks), os.cpu_count() or 4)

    logger.info(f"开始双策略处理 {len(entity_chunks)} 个实体分块，使用 {max_workers} 个线程")

    # 统计普通实体和题目实体数量
    total_normal = 0
    total_problem = 0
    for chunk_name, entities in entity_chunks.items():
        normal_count = sum(1 for e in entities if not e.get("is_problem_extracted", False))
        problem_count = sum(1 for e in entities if e.get("is_problem_extracted", False))
        total_normal += normal_count
        total_problem += problem_count
        logger.info(f"  分块 '{chunk_name}': 普通实体 {normal_count}, 题目实体 {problem_count}")

    logger.info(f"总计: 普通实体 {total_normal}, 题目实体 {total_problem}")

    def process_chunk_in_thread(chunk_name: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """在线程中统一处理分块 - 混合处理普通实体和题目实体"""
        try:
            # 分离普通实体和题目实体，并按顺序排列（普通实体在前，题目实体在后）
            normal_entities = [e for e in entities if not e.get("is_problem_extracted", False)]
            problem_entities = [e for e in entities if e.get("is_problem_extracted", False)]
            sorted_entities = normal_entities + problem_entities

            # 统一调用 process_entity_chunk_with_llm 进行混合处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    process_entity_chunk_with_llm(
                        chunk_name=chunk_name,
                        entities=sorted_entities,
                        llm_func=llm_func,
                        entity_types_dim1=entity_types_dim1,
                        entity_types_dim2=entity_types_dim2,
                        topk=topk,
                        max_concurrent=max_concurrent,
                        # 普通实体使用LLM策略：禁用相似度自动分流
                        high_similarity_threshold=1.0,  # 永不自动合并
                        low_similarity_threshold=0.0,   # 永不自动跳过
                        # 题目实体使用快速策略：基于相似度阈值判断
                        problem_similarity_threshold=0.8
                    )
                )
                # 添加统计信息
                result["normal_entity_count"] = len(normal_entities)
                result["problem_entity_count"] = len(problem_entities)
                return result
            finally:
                loop.close()

        except Exception as exc:
            logger.error(f"分块 '{chunk_name}' 统一处理失败: {exc}")
            return {
                "chunk_name": chunk_name,
                "status": "failed",
                "error": str(exc),
                "merged_entities": entities,  # 失败时返回原始实体
                "processed_at": time.time()
            }

    results = {}

    # 使用 ThreadPoolExecutor 并行处理分块
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        for chunk_name, entities in entity_chunks.items():
            future = executor.submit(process_chunk_in_thread, chunk_name, entities)
            future_to_chunk[future] = chunk_name
            logger.info(f"提交分块 '{chunk_name}' 到线程池 (双策略)")

        # 收集处理结果
        for future in as_completed(future_to_chunk):
            chunk_name = future_to_chunk[future]
            try:
                result = future.result()
                results[chunk_name] = result
                if result.get("status") == "processed":
                    logger.info(f"分块 '{chunk_name}' 双策略处理完成: "
                               f"普通 {result.get('normal_entity_count', 0)}, "
                               f"题目 {result.get('problem_entity_count', 0)}, "
                               f"合并后 {len(result.get('merged_entities', []))}")
            except Exception as exc:
                logger.error(f"分块 '{chunk_name}' 处理失败: {exc}")
                results[chunk_name] = {
                    "chunk_name": chunk_name,
                    "status": "failed",
                    "error": str(exc),
                    "merged_entities": [],
                    "processed_at": time.time()
                }

    # 统计处理结果
    successful = sum(1 for r in results.values() if r.get('status') == 'processed')
    total_merged = sum(len(r.get('merged_entities', [])) for r in results.values())
    logger.info(f"双策略处理完成: {successful}/{len(entity_chunks)} 分块成功, 最终实体数: {total_merged}")

    return results


def chunk_entities_by_dim1_type(entities: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按dim1类型将实体数据分块
    
    Args:
        entities: 实体列表
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: 按dim1类型分块的字典
    """
    chunks = {}
    
    for entity in entities:
        dim1_type = entity.get("entity_type_dim1", "UNKNOWN")
        
        if dim1_type not in chunks:
            chunks[dim1_type] = []
        
        chunks[dim1_type].append(entity)
    
    return chunks

def print_chunk_summary(chunks: Dict[str, List[Dict[str, Any]]]):
    """打印分块摘要"""
    print("\n" + "="*60)
    print("实体分块摘要")
    print("="*60)

    for dim1_type, chunk_entities in chunks.items():
        print(f"\n类型: {dim1_type}")
        print(f"  实体数量: {len(chunk_entities)}")

        # 统计子类型分布
        dim2_counts = {}
        for entity in chunk_entities:
            dim2_type = entity.get("entity_type_dim2", "UNKNOWN")
            dim2_counts[dim2_type] = dim2_counts.get(dim2_type, 0) + 1

        for dim2_type, count in sorted(dim2_counts.items()):
            print(f"    {dim2_type}: {count}")

    print("="*60)


# ==================== LLM调用函数 ====================

def extract_entity_name(entity_id: str) -> str:
    """从实体ID中提取纯名称（去掉随机ID部分）"""
    # 格式: 实体名<随机ID> 或 实体名
    if '<' in entity_id and entity_id.endswith('>'):
        return entity_id[:entity_id.rfind('<')]
    return entity_id


def extract_problem_id(entity_id: str) -> str:
    """从实体ID中提取题目ID（前缀部分，空格分隔）"""
    if ' ' in entity_id:
        return entity_id.split(' ')[0]
    return ""


def embed_list_to_base64(embedding: list) -> str:
    """将embedding列表转为base64字符串，减少存储空间"""
    import numpy as np
    import base64
    arr = np.array(embedding, dtype=np.float32)
    return base64.b64encode(arr.tobytes()).decode('ascii')


def embed_base64_to_list(compressed: str) -> list:
    """将base64字符串转为embedding列表"""
    import numpy as np
    import base64
    data = base64.b64decode(compressed.encode('ascii'))
    arr = np.frombuffer(data, dtype=np.float32)
    return arr.tolist()


def generate_entity_id(name: str) -> str:
    """为实体名称生成带随机ID的实体ID"""
    import random
    import string
    random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
    return f"{name}<{random_suffix}>"


def parse_llm_json_response(response: str, default_result: Dict) -> Dict:
    """解析LLM返回的JSON，失败时返回默认值"""
    try:
        # 尝试提取JSON块
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # 尝试直接解析
        return json.loads(response.strip())
    except Exception as e:
        logger.warning(f"解析LLM JSON响应失败: {e}, 使用默认值")
        return default_result


async def call_llm_with_retry(
    llm_func: Callable,
    prompt: str,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> str:
    """
    带重试机制的LLM调用

    Args:
        llm_func: LLM调用函数
        prompt: 提示词
        max_retries: 最大重试次数
        retry_delay: 重试间隔（秒）

    Returns:
        str: LLM响应
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            response = await llm_func(prompt)
            return response
        except Exception as e:
            last_error = e
            error_str = str(e)
            # 判断是否是可重试的错误（网络错误、502、504、超时等）
            # 明确排除503错误，因为503表示服务不可用，立即失败更合理
            is_retryable = any(keyword in error_str.lower() for keyword in [
                '502', '504', 'bad gateway', 'timeout', 'connection',
                'network', 'reset', 'refused', 'unavailable'
            ]) and '503' not in error_str.lower()

            if is_retryable and attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # 指数退避
                logger.warning(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {error_str[:100]}..., {wait_time}秒后重试")
                await asyncio.sleep(wait_time)
            else:
                # 503错误或其他不可重试错误，立即失败
                if '503' in error_str.lower():
                    logger.error(f"LLM服务不可用 (503错误): {error_str[:100]}，跳过重试")
                raise

    raise last_error


async def call_entity_evaluation_llm(
    current_entity: Dict[str, Any],
    similar_entities: List[Dict[str, Any]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None
) -> Dict[str, Any]:
    """
    调用LLM评估单个实体

    Args:
        current_entity: 当前实体
        similar_entities: 相似实体列表
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表

    Returns:
        {
            "should_delete": bool,
            "delete_reason": str | None,
            "should_merge": bool,
            "merge_target": str | None,
            "merge_reason": str | None,
            "corrected_name": str,
            "corrected_type_dim1": str,
            "corrected_type_dim2": str
        }
    """
    entity_types_dim1 = entity_types_dim1 or DEFAULT_ENTITY_TYPES_DIM1
    entity_types_dim2 = entity_types_dim2 or DEFAULT_ENTITY_TYPES_DIM2

    entity_id = current_entity.get("entity_id", "")

    # 构建输入JSON - 使用完整ID（保留随机ID）
    current_entity_json = json.dumps({
        "id": entity_id,
        "type_dim1": current_entity.get("entity_type_dim1", "UNKNOWN"),
        "type_dim2": current_entity.get("entity_type_dim2", "UNKNOWN"),
        "description": current_entity.get("description", "")
    }, ensure_ascii=False, indent=2)

    # 相似实体保留完整ID
    similar_entities_json = json.dumps([
        {
            "id": e.get("entity_id", ""),
            "type_dim1": e.get("entity_type_dim1", "UNKNOWN"),
            "type_dim2": e.get("entity_type_dim2", "UNKNOWN"),
            "description": e.get("description", "")
        }
        for e in similar_entities
    ], ensure_ascii=False, indent=2)

    # 构建prompt
    prompt = PROMPTS["entity_merge_evaluation"].format(
        entity_types_dim1=", ".join(entity_types_dim1),
        entity_types_dim2=", ".join(entity_types_dim2),
        current_entity=current_entity_json,
        similar_entities=similar_entities_json
    )

    # 默认结果
    default_result = {
        "should_delete": False,
        "should_merge": False,
        "merge_target": None
    }

    try:
        # 调用LLM（带重试机制，由call_llm_with_retry自己管理超时）
        response = await call_llm_with_retry(llm_func, prompt)
        result = parse_llm_json_response(response, default_result)

        # 验证merge_target是否在similar_entities中（使用完整ID比较）
        if result.get("should_merge") and result.get("merge_target"):
            valid_targets = {e.get("entity_id", "") for e in similar_entities}
            if result["merge_target"] not in valid_targets:
                logger.warning(f"实体 {entity_id} 的合并目标 {result['merge_target']} 不在相似实体列表中，取消合并")
                result["should_merge"] = False
                result["merge_target"] = None

        return result

    except asyncio.TimeoutError:
        logger.error(f"LLM评估实体 {entity_id} 超时，返回默认结果")
        return default_result
    except Exception as e:
        logger.error(f"LLM评估实体 {entity_id} 失败: {e}")
        return default_result


async def call_entity_group_merge_llm(
    entities: List[Dict[str, Any]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None
) -> Dict[str, Any]:
    """
    调用LLM合并实体组

    Args:
        entities: 待合并的实体列表
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表

    Returns:
        {
            "final_name": str,
            "final_type_dim1": str,
            "final_type_dim2": str,
            "final_description": str
        }
    """
    entity_types_dim1 = entity_types_dim1 or DEFAULT_ENTITY_TYPES_DIM1
    entity_types_dim2 = entity_types_dim2 or DEFAULT_ENTITY_TYPES_DIM2

    # 构建输入JSON - 保留完整ID（带随机ID）
    entities_json = json.dumps([
        {
            "id": e.get("entity_id", ""),
            "type_dim1": e.get("entity_type_dim1", "UNKNOWN"),
            "type_dim2": e.get("entity_type_dim2", "UNKNOWN"),
            "description": e.get("description", "")
        }
        for e in entities
    ], ensure_ascii=False, indent=2)

    prompt = PROMPTS["entity_group_merge"].format(
        entity_types_dim1=", ".join(entity_types_dim1),
        entity_types_dim2=", ".join(entity_types_dim2),
        entities_list=entities_json
    )

    # 默认结果：使用第一个实体的信息，描述用SEP拼接
    default_result = {
        "final_name": entities[0].get("entity_id", ""),
        "final_type_dim1": entities[0].get("entity_type_dim1", "其他"),
        "final_type_dim2": entities[0].get("entity_type_dim2", "概念"),
        "final_description": "<SEP>".join(e.get("description", "") for e in entities if e.get("description"))
    }

    try:
        # 调用LLM（带重试机制，由call_llm_with_retry自己管理超时）
        response = await call_llm_with_retry(llm_func, prompt)
        result = parse_llm_json_response(response, default_result)

        # 为最终名称生成新的随机ID
        final_name = result.get("final_name", entities[0].get("entity_id", ""))
        result["final_name"] = generate_entity_id(final_name)

        return result
    except asyncio.TimeoutError:
        logger.error(f"LLM合并实体组超时，返回默认结果")
        return default_result
    except Exception as e:
        logger.error(f"LLM合并实体组失败: {e}")
        return default_result


async def call_entity_reclassify_llm(
    entity: Dict[str, Any],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None
) -> Dict[str, Any]:
    """
    调用LLM对UNKNOWN类型实体进行重分类

    Args:
        entity: 实体数据
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表

    Returns:
        {
            "should_delete": bool,
            "delete_reason": str | None,
            "corrected_name": str,
            "type_dim1": str,
            "type_dim2": str
        }
    """
    entity_types_dim1 = entity_types_dim1 or DEFAULT_ENTITY_TYPES_DIM1
    entity_types_dim2 = entity_types_dim2 or DEFAULT_ENTITY_TYPES_DIM2

    entity_id = entity.get("entity_id", "")
    entity_desc = entity.get("description", "")
    # 提取纯名称（去掉随机ID）发送给LLM
    clean_entity_name = extract_entity_name(entity_id)

    prompt = PROMPTS["entity_reclassify"].format(
        entity_types_dim1=", ".join(entity_types_dim1),
        entity_types_dim2=", ".join(entity_types_dim2),
        entity_name=clean_entity_name,
        entity_description=entity_desc
    )

    default_result = {
        "should_delete": False,
        "corrected_name": entity_id,
        "type_dim1": "其他",
        "type_dim2": "概念"
    }

    try:
        response = await call_llm_with_retry(llm_func, prompt)

        result = parse_llm_json_response(response, default_result)

        # 为修正后的名称生成新的随机ID
        corrected_name = result.get("corrected_name", entity_id)
        result["corrected_name"] = generate_entity_id(corrected_name)

        return result
    except asyncio.TimeoutError:
        logger.error(f"LLM重分类实体 {entity_id} 超时，返回默认结果")
        return default_result
    except Exception as e:
        logger.error(f"LLM重分类实体 {entity_id} 失败: {e}")
        return default_result


async def reclassify_unknown_entities(
    entities: List[Dict[str, Any]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None,
    max_concurrent: int = None
) -> Tuple[List[Dict[str, Any]], int]:
    """
    预处理UNKNOWN类型的实体，使用LLM进行重分类

    Args:
        entities: 实体列表
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表
        max_concurrent: 最大并发数

    Returns:
        Tuple[List[Dict[str, Any]], int]: (处理后的实体列表, 删除的实体数)
    """
    max_concurrent = max_concurrent or DEFAULT_ENTITY_MERGE_MAX_ASYNC

    # 分离UNKNOWN实体和正常实体
    unknown_entities = []
    normal_entities = []

    for entity in entities:
        dim1 = entity.get("entity_type_dim1", "UNKNOWN")
        dim2 = entity.get("entity_type_dim2", "UNKNOWN")
        if dim1 == "UNKNOWN" or dim2 == "UNKNOWN" or dim1 == "" or dim2 == "":
            unknown_entities.append(entity)
        else:
            normal_entities.append(entity)

    if not unknown_entities:
        logger.info("没有UNKNOWN类型的实体需要重分类")
        return entities, 0

    logger.info(f"开始重分类 {len(unknown_entities)} 个UNKNOWN类型实体")

    # 并发处理UNKNOWN实体
    semaphore = asyncio.Semaphore(max_concurrent)
    deleted_count = 0
    reclassified_entities = []

    async def process_single(entity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        nonlocal deleted_count
        async with semaphore:
            entity_id = entity.get("entity_id", "")

            result = await call_entity_reclassify_llm(
                entity=entity,
                llm_func=llm_func,
                entity_types_dim1=entity_types_dim1,
                entity_types_dim2=entity_types_dim2
            )

            if result.get("should_delete"):
                logger.info(f"  删除实体: {entity_id}")
                deleted_count += 1
                return None

            # 更新实体信息
            updated_entity = entity.copy()
            updated_entity["entity_id"] = result.get("corrected_name", entity_id)
            updated_entity["entity_type_dim1"] = result.get("type_dim1", "其他")
            updated_entity["entity_type_dim2"] = result.get("type_dim2", "概念")

            if updated_entity["entity_id"] != entity_id or updated_entity["entity_type_dim1"] != entity.get("entity_type_dim1"):
                logger.info(f"  重分类: {entity_id} -> {updated_entity['entity_id']}, 类型: {updated_entity['entity_type_dim1']}/{updated_entity['entity_type_dim2']}")

            return updated_entity

    # 并发执行
    tasks = [process_single(entity) for entity in unknown_entities]
    results = await asyncio.gather(*tasks)

    # 收集非None结果
    for result in results:
        if result is not None:
            reclassified_entities.append(result)

    # 合并结果
    final_entities = normal_entities + reclassified_entities

    logger.info(f"UNKNOWN实体重分类完成:")
    logger.info(f"  - 原UNKNOWN实体数: {len(unknown_entities)}")
    logger.info(f"  - 删除实体数: {deleted_count}")
    logger.info(f"  - 重分类实体数: {len(reclassified_entities)}")
    logger.info(f"  - 最终实体数: {len(final_entities)}")

    return final_entities, deleted_count


def merge_entity_group(group: List[str], entities_dict: Dict[str, Dict[str, Any]], chunk_name: str) -> Dict[str, Any]:
    """
    合并实体组，根据实体数量采用不同的合并策略（同步版本）

    Args:
        group: 实体ID列表
        entities_dict: 实体字典，ID -> 实体数据
        chunk_name: 分块名称（用于日志）

    Returns:
        Dict[str, Any]: 合并后的实体
    """
    from collections import Counter

    # 收集组内所有实体
    group_entities = [entities_dict[entity_id] for entity_id in group if entity_id in entities_dict]

    if not group_entities:
        logger.warning(f"实体组 {group} 在entities_dict中未找到，返回空实体")
        return {}

    # 选择主实体（第一个实体作为基础）
    main_entity = group_entities[0].copy()
    primary_entity_id = group_entities[0].get('entity_id', '')

    logger.info(f"开始合并 {len(group_entities)} 个实体，主实体: {primary_entity_id}")

    # 合并实体类型
    dim1_types = [e.get('entity_type_dim1', '') for e in group_entities if e.get('entity_type_dim1')]
    dim2_types = [e.get('entity_type_dim2', '') for e in group_entities if e.get('entity_type_dim2')]

    # 选择最常见的类型作为主要类型
    if dim1_types:
        dim1_counter = Counter(dim1_types)
        main_entity['entity_type_dim1'] = dim1_counter.most_common(1)[0][0]

    if dim2_types:
        dim2_counter = Counter(dim2_types)
        main_entity['entity_type_dim2'] = dim2_counter.most_common(1)[0][0]

    # 收集所有描述
    descriptions = []
    for entity in group_entities:
        desc = entity.get('description', '').strip()
        if desc:
            descriptions.append(desc)

    # 收集源信息和文件路径
    source_ids = []
    file_paths = []
    for entity in group_entities:
        source_id = entity.get('source_id', '').strip()
        if source_id:
            source_ids.extend(source_id.split('<SEP>') if '<SEP>' in source_id else [source_id])

        file_path = entity.get('file_path', '').strip()
        if file_path:
            file_paths.extend(file_path.split('<SEP>') if '<SEP>' in file_path else [file_path])

    # 去重收集到的信息
    source_ids = list(dict.fromkeys(source_ids))
    file_paths = list(dict.fromkeys(file_paths))

    # 合并描述策略
    GRAPH_FIELD_SEP = "<SEP>"
    FORCE_LLM_THRESHOLD = 4  # 超过4个描述使用LLM

    if len(descriptions) <= FORCE_LLM_THRESHOLD:
        # 少量描述直接拼接
        main_entity['description'] = GRAPH_FIELD_SEP.join(descriptions) if descriptions else ""
        logger.info(f"  → 直接拼接 {len(descriptions)} 个描述")
    else:
        # 大量描述使用LLM生成
        try:
            # 导入LLM摘要服务
            from llm_summary_service import LLMSummaryService
            llm_service = LLMSummaryService()

            entity_name = primary_entity_id or f"Entity_{chunk_name}"
            logger.info(f"  → 使用LLM合并 {len(descriptions)} 个描述")

            # 同步调用LLM摘要服务
            async def _summarize_descriptions():
                return await llm_service.summarize_descriptions(
                    description_type="Entity",
                    description_name=entity_name,
                    description_list=descriptions,
                    separator=GRAPH_FIELD_SEP,
                    summary_length="One Paragraphs",
                    language="zh-CN"
                )

            final_description = asyncio.run(_summarize_descriptions())
            main_entity['description'] = final_description
            logger.info(f"    LLM生成的描述: {final_description[:100]}...")

        except Exception as e:
            logger.warning(f"LLM描述生成失败: {e}，使用原始描述拼接")
            main_entity['description'] = GRAPH_FIELD_SEP.join(descriptions)

    # 合并其他字段
    if source_ids:
        main_entity['source_id'] = GRAPH_FIELD_SEP.join(source_ids)

    if file_paths:
        main_entity['file_path'] = GRAPH_FIELD_SEP.join(file_paths)

    # 更新创建时间
    main_entity['created_at'] = group_entities[0].get('created_at', int(time.time()))

    # 记录合并信息
    main_entity['merged_from'] = group
    main_entity['merge_count'] = len(group_entities)
    main_entity['merge_timestamp'] = int(time.time())

    # 为合并后的实体生成新的嵌入向量
    logger.info(f"  → 开始为合并实体生成新的嵌入向量...")
    try:
        # 构造用于生成嵌入的文本内容
        entity_name_for_embed = primary_entity_id or f"Entity_{chunk_name}"
        description_for_embed = main_entity.get('description', '')
        content_to_embed = f"{entity_name_for_embed}\n{description_for_embed}".strip()

        if content_to_embed:
            # 同步版本：使用主实体向量作为备用
            # 注意：同步版本无法直接调用异步embedding_func，建议使用异步版本
            logger.info(f"    同步版本：建议使用异步版本获取新嵌入向量")
            if group_entities[0].get('embedding') is not None:
                main_entity['embedding'] = group_entities[0]['embedding']
                logger.info(f"    使用主实体嵌入向量作为备用")
        else:
            logger.warning(f"    合并实体无内容用于生成嵌入")
            if group_entities[0].get('embedding') is not None:
                main_entity['embedding'] = group_entities[0]['embedding']

    except Exception as e:
        logger.error(f"    处理嵌入向量失败: {e}，使用主实体向量")
        if group_entities[0].get('embedding') is not None:
            main_entity['embedding'] = group_entities[0]['embedding']

    logger.info(f"  → 合并完成: {primary_entity_id}")
    logger.info(f"    类型: {main_entity.get('entity_type_dim1', 'UNKNOWN')}/{main_entity.get('entity_type_dim2', 'UNKNOWN')}")
    logger.info(f"    描述长度: {len(main_entity.get('description', ''))}")
    logger.info(f"    源数量: {len(source_ids)}, 文件数量: {len(file_paths)}")
    logger.info(f"    嵌入向量: {'✓' if main_entity.get('embedding') is not None else '✗'}")

    return main_entity


async def merge_entity_group_async(group: List[str], entities_dict: Dict[str, Dict[str, Any]], chunk_name: str) -> Dict[str, Any]:
    """
    合并实体组，根据实体数量采用不同的合并策略（异步版本）
    
    Args:
        group: 实体ID列表
        entities_dict: 实体字典，ID -> 实体数据
        chunk_name: 分块名称（用于日志）
        
    Returns:
        Dict[str, Any]: 合并后的实体
    """
    from collections import Counter
    
    # 收集组内所有实体
    group_entities = [entities_dict[entity_id] for entity_id in group if entity_id in entities_dict]
    
    if not group_entities:
        logger.warning(f"实体组 {group} 在entities_dict中未找到，返回空实体")
        return {}
    
    # 选择主实体（第一个实体作为基础）
    main_entity = group_entities[0].copy()
    primary_entity_id = group_entities[0].get('entity_id', '')
    
    logger.info(f"开始合并 {len(group_entities)} 个实体，主实体: {primary_entity_id}")
    
    # 合并实体类型
    dim1_types = [e.get('entity_type_dim1', '') for e in group_entities if e.get('entity_type_dim1')]
    dim2_types = [e.get('entity_type_dim2', '') for e in group_entities if e.get('entity_type_dim2')]
    
    # 选择最常见的类型作为主要类型
    if dim1_types:
        dim1_counter = Counter(dim1_types)
        main_entity['entity_type_dim1'] = dim1_counter.most_common(1)[0][0]
    
    if dim2_types:
        dim2_counter = Counter(dim2_types)
        main_entity['entity_type_dim2'] = dim2_counter.most_common(1)[0][0]
    
    # 收集所有描述
    descriptions = []
    for entity in group_entities:
        desc = entity.get('description', '').strip()
        if desc:
            descriptions.append(desc)
    
    # 收集源信息和文件路径
    source_ids = []
    file_paths = []
    for entity in group_entities:
        source_id = entity.get('source_id', '').strip()
        if source_id:
            source_ids.extend(source_id.split('<SEP>') if '<SEP>' in source_id else [source_id])
        
        file_path = entity.get('file_path', '').strip()
        if file_path:
            file_paths.extend(file_path.split('<SEP>') if '<SEP>' in file_path else [file_path])
    
    # 去重收集到的信息
    source_ids = list(dict.fromkeys(source_ids))
    file_paths = list(dict.fromkeys(file_paths))
    
    # 合并描述策略
    GRAPH_FIELD_SEP = "<SEP>"
    FORCE_LLM_THRESHOLD = 4  # 超过4个描述使用LLM
    
    if len(descriptions) <= FORCE_LLM_THRESHOLD:
        # 少量描述直接拼接
        main_entity['description'] = GRAPH_FIELD_SEP.join(descriptions) if descriptions else ""
        logger.info(f"  → 直接拼接 {len(descriptions)} 个描述")
    else:
        # 大量描述使用LLM生成
        try:
            # 导入LLM摘要服务
            from llm_summary_service import LLMSummaryService
            llm_service = LLMSummaryService()
            
            entity_name = primary_entity_id or f"Entity_{chunk_name}"
            logger.info(f"  → 使用LLM合并 {len(descriptions)} 个描述")
            
            # 异步调用LLM摘要服务
            final_description = await llm_service.summarize_descriptions(
                description_type="Entity",
                description_name=entity_name,
                description_list=descriptions,
                separator=GRAPH_FIELD_SEP,
                summary_length="One Paragraphs",
                language="zh-CN"
            )
            main_entity['description'] = final_description
            logger.info(f"    LLM生成的描述: {final_description[:100]}...")
            
        except Exception as e:
            logger.warning(f"LLM描述生成失败: {e}，使用原始描述拼接")
            main_entity['description'] = GRAPH_FIELD_SEP.join(descriptions)
    
    # 合并其他字段
    if source_ids:
        main_entity['source_id'] = GRAPH_FIELD_SEP.join(source_ids)

    if file_paths:
        main_entity['file_path'] = GRAPH_FIELD_SEP.join(file_paths)

    # 更新创建时间
    main_entity['created_at'] = group_entities[0].get('created_at', int(time.time()))

    # 记录合并信息
    main_entity['merged_from'] = group
    main_entity['merge_count'] = len(group_entities)
    main_entity['merge_timestamp'] = int(time.time())

    # 为合并后的实体生成新的嵌入向量
    logger.info(f"  → 开始为合并实体生成新的嵌入向量...")
    try:
        # 构造用于生成嵌入的文本内容
        entity_name_for_embed = primary_entity_id or f"Entity_{chunk_name}"
        description_for_embed = main_entity.get('description', '')
        content_to_embed = f"{entity_name_for_embed}\n{description_for_embed}".strip()

        if content_to_embed:
            # 异步版本：可以使用embedding_func
            # 注意：这里需要从外部传递embedding_func，或者使用全局变量
            # 暂时使用主实体向量作为备用
            logger.info(f"    异步版本：建议从外部传递embedding_func参数")
            if group_entities[0].get('embedding') is not None:
                main_entity['embedding'] = group_entities[0]['embedding']
                logger.info(f"    使用主实体嵌入向量作为备用")
        else:
            logger.warning(f"    合并实体无内容用于生成嵌入")
            if group_entities[0].get('embedding') is not None:
                main_entity['embedding'] = group_entities[0]['embedding']

    except Exception as e:
        logger.error(f"    处理嵌入向量失败: {e}，使用主实体向量")
        if group_entities[0].get('embedding') is not None:
            main_entity['embedding'] = group_entities[0]['embedding']

    logger.info(f"  → 合并完成: {primary_entity_id}")
    logger.info(f"    类型: {main_entity.get('entity_type_dim1', 'UNKNOWN')}/{main_entity.get('entity_type_dim2', 'UNKNOWN')}")
    logger.info(f"    描述长度: {len(main_entity.get('description', ''))}")
    logger.info(f"    源数量: {len(source_ids)}, 文件数量: {len(file_paths)}")
    logger.info(f"    嵌入向量: {'✓' if main_entity.get('embedding') is not None else '✗'}")

    return main_entity


def process_entity_chunk(chunk_name: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    处理单个实体分块，包含向量检索优化
    
    Args:
        chunk_name: 分块名称（dim1类型）
        entities: 该分块中的实体列表
        
    Returns:
        Dict[str, Any]: 处理结果，包含分块名称和状态信息
    """
    # 导入轻量级向量检索库
    import hnswlib
    
    # 创建向量检索实例用于实体对齐
    entities_with_embeddings = [e for e in entities if e.get("embedding") is not None]
    
    if not entities_with_embeddings:
        logger.warning(f"分块 '{chunk_name}' 没有包含嵌入向量的实体，跳过处理")
        return {
            "chunk_name": chunk_name,
            "entity_count": len(entities),
            "status": "skipped",
            "merged_entities": entities,
            "map": {}
        }
    
    # 初始化hnsw向量索引
    embedding_dim = len(entities_with_embeddings[0]["embedding"])
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.init_index(max_elements=len(entities_with_embeddings), ef_construction=200, M=16)
    index.set_ef(50)

    #Dict[str, str] 实体id: 实体id
    #类似并查集结构，维护相似实体合并
    entities_map = dict()
    entities_dict = dict()
    for entity in entities:
        id = entity.get('entity_id')
        entities_map[id] = id
        entities_dict[id] = entity
    logger.info(f"分块 '{chunk_name}' 向量索引初始化完成，包含 {len(entities_with_embeddings)} 个实体")
    
    # 完成实体对齐
    similarity_threshold = 0.8  # 相似度阈值，可根据需要调整
    
    for idx, entity in enumerate(entities):
        #找到最相似实体
        # 查询当前实体之前的所有实体（避免自己匹配自己）
        if idx > 0 and entity.get("embedding") is not None:
            # 使用hnsw向量索引查询最相似的实体
            # 注意：只查询已插入的实体，避免匹配自己
            candidates, distances = index.knn_query(entity["embedding"], k=min(idx, 3))
            if len(candidates) > 0:
                # 获取最相似的实体
                best_match_idx = candidates[0][0]
                best_distance = distances[0][0]
                
                # 将距离转换为相似度（余弦距离转相似度）
                similarity = 1.0 - best_distance
                
                #判断相似度，大于阈值就在mapping记录合并
                if similarity > similarity_threshold:
                    best_match_entity = entities[best_match_idx]
                    current_entity_id = entity.get("entity_id")
                    best_match_entity_id = best_match_entity.get("entity_id")
                    
                    if current_entity_id and best_match_entity_id:
                        # 在并查集中记录合并关系
                        # 将当前实体的根节点指向最相似实体的根节点
                        entities_map[current_entity_id] = best_match_entity_id
        
        #最后，插入本节点到索引
        if entity.get("embedding") is not None:
            index.add_items(entity["embedding"], idx)

    #现在mapping并查集还没有经过路径压缩，进行一个遍历完成全部路径压缩。然后按相同合并目标的节点分块放进一个list,方便后续实际执行合并
    
    # 定义并查集的find函数，用于路径压缩
    def find(entity_id):
        """查找实体的根节点，同时进行路径压缩"""
        if entities_map[entity_id] != entity_id:
            entities_map[entity_id] = find(entities_map[entity_id])
        return entities_map[entity_id]
    
    # 进行路径压缩
    for entity_id in entities_map:
        entities_map[entity_id] = find(entity_id)
    
    # 按相同合并目标的节点分块放进一个list,方便后续实际执行合并
    merge_groups = {}
    for entity_id, root_id in entities_map.items():
        if root_id not in merge_groups:
            merge_groups[root_id] = []
        merge_groups[root_id].append(entity_id)
    
    # 实际执行实体合并
    merged_entities = []
    for root_id, group in merge_groups.items():
        #logger.info(f"  - 合并组 ({root_id}): {len(group)} 个实体")
        
        if len(group) == 1:
            # 单个实体直接保留
            merged_entities.append(entities_dict[group[0]])
            #logger.info(f"    → 单个实体直接保留: {group[0]}")
        else:
            # 多个实体需要合并
            merged_entity = merge_entity_group(group, entities_dict, chunk_name)
            merged_entities.append(merged_entity)
            logger.info(f"    → 合并为: {merged_entity.get('entity_id', 'unknown')} (包含{len(group)}个实体)")
    
    logger.info(f"分块 '{chunk_name}' 实体合并完成:")
    logger.info(f"  - 总实体数: {len(entities_with_embeddings)}")
    logger.info(f"  - 合并组: {len(merge_groups)}")
    logger.info(f"  - 合并后实体数: {len(merged_entities)}")

    result = {
        "chunk_name": chunk_name,
        "entity_count": len(entities),
        "status": "processed",
        "merged_entities": merged_entities,
        "map": entities_map
    }
    
    return result


def process_entity_chunks_multithreaded(entity_chunks: Dict[str, List[Dict[str, Any]]], max_workers: int = None) -> Dict[str, Dict[str, Any]]:
    """
    多线程处理实体分块
    
    Args:
        entity_chunks: 按dim1类型分块的实体字典
        max_workers: 最大线程数，默认为CPU核心数
        
    Returns:
        Dict[str, Dict[str, Any]]: 每个分块的处理结果
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = min(len(entity_chunks), os.cpu_count() or 4)
    
    logger.info(f"开始多线程处理 {len(entity_chunks)} 个实体分块，使用 {max_workers} 个线程")
    
    results = {}
    
    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有分块处理任务
        future_to_chunk = {}
        for chunk_name, entities in entity_chunks.items():
            future = executor.submit(process_entity_chunk, chunk_name, entities)
            future_to_chunk[future] = chunk_name
            logger.info(f"提交分块 '{chunk_name}' 到线程池，包含 {len(entities)} 个实体")
        
        # 收集处理结果
        for future in as_completed(future_to_chunk):
            chunk_name = future_to_chunk[future]
            try:
                result = future.result()
                results[chunk_name] = result
                #logger.info(f"分块 '{chunk_name}' 处理完成: {result}")
            except Exception as exc:
                logger.error(f"分块 '{chunk_name}' 处理失败: {exc}")
                results[chunk_name] = {
                    "chunk_name": chunk_name,
                    "status": "failed",
                    "error": str(exc),
                    "processed_at": time.time()
                }
    
    logger.info(f"多线程处理完成，成功处理 {sum(1 for r in results.values() if r.get('status') == 'processed')} 个分块")

    return results


async def process_entity_chunk_with_llm(
    chunk_name: str,
    entities: List[Dict[str, Any]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None,
    topk: int = None,
    max_concurrent: int = None,
    high_similarity_threshold: float = 0.85,
    low_similarity_threshold: float = 0.6,
    problem_similarity_threshold: float = 0.8,
    small_merge_threshold: int = 4
) -> Dict[str, Any]:
    """
    使用LLM增强的实体分块处理（统一处理普通实体和题目实体）

    处理流程：
    1. 先排序：普通实体在前，题目实体在后（共用同一个HNSW索引）
    2. 普通实体：HNSW检索 → LLM评估是否合并
    3. 题目实体：HNSW检索 → 相似度阈值判断是否合并（不用LLM评估）
    4. 合并时都用LLM合并描述

    Args:
        chunk_name: 分块名称（dim1类型）
        entities: 该分块中的实体列表（包含普通实体和题目实体）
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表
        topk: 检索相似实体的数量
        max_concurrent: 最大并发数
        high_similarity_threshold: 普通实体高相似度阈值（>=此值直接合并）
        low_similarity_threshold: 普通实体低相似度阈值（<=此值直接跳过）
        problem_similarity_threshold: 题目实体相似度阈值（>=此值直接合并）

    Returns:
        Dict[str, Any]: 处理结果
    """
    import hnswlib

    topk = topk or DEFAULT_ENTITY_MERGE_TOPK
    max_concurrent = max_concurrent or DEFAULT_ENTITY_MERGE_MAX_ASYNC
    entity_types_dim1 = entity_types_dim1 or DEFAULT_ENTITY_TYPES_DIM1
    entity_types_dim2 = entity_types_dim2 or DEFAULT_ENTITY_TYPES_DIM2

    # ==================== 排序：普通实体在前，题目实体在后 ====================
    normal_entities = [e for e in entities if not e.get("is_problem_extracted", False)]
    problem_entities = [e for e in entities if e.get("is_problem_extracted", False)]
    sorted_entities = normal_entities + problem_entities

    logger.info(f"分块 '{chunk_name}' 实体排序: 普通实体 {len(normal_entities)} 个在前, 题目实体 {len(problem_entities)} 个在后")

    # 过滤有嵌入向量的实体（保持排序）
    entities_with_embeddings = [e for e in sorted_entities if e.get("embedding") is not None]

    if not entities_with_embeddings:
        logger.warning(f"分块 '{chunk_name}' 没有包含嵌入向量的实体，跳过处理")
        return {
            "chunk_name": chunk_name,
            "entity_count": len(entities),
            "status": "skipped",
            "merged_entities": entities,
            "map": {},
            "deleted_count": 0
        }

    # 初始化HNSW索引
    embedding_dim = len(entities_with_embeddings[0]["embedding"])
    index = hnswlib.Index(space='cosine', dim=embedding_dim)
    index.init_index(max_elements=len(entities_with_embeddings), ef_construction=200, M=16)
    index.set_ef(50)

    # 并查集和数据结构
    entities_map = {}
    entities_dict = {}
    deleted_entities = set()  # 记录被删除的实体

    for entity in entities:
        entity_id = entity.get('entity_id')
        entities_map[entity_id] = entity_id
        entities_dict[entity_id] = entity

    logger.info(f"分块 '{chunk_name}' 开始LLM增强处理，包含 {len(entities_with_embeddings)} 个实体")

    # ==================== 阶段1：HNSW顺序添加 + 智能分流处理 ====================
    logger.info(f"  阶段1：HNSW顺序添加 + 智能分流处理启动...")
    logger.info(f"  普通实体相似度阈值: 高>{high_similarity_threshold}, 低<{low_similarity_threshold}")
    logger.info(f"  题目实体相似度阈值: >={problem_similarity_threshold}")

    # 信号量控制并发
    semaphore = asyncio.Semaphore(max_concurrent)

    # 存储所有LLM任务
    llm_tasks = []

    # 记录已插入索引的实体索引（用于HNSW正确性）
    inserted_indices = []

    # 记录实体处理顺序（用于并查集）
    processing_order = []

    # 性能统计
    auto_merged_count = 0      # 自动合并数
    auto_skipped_count = 0     # 自动跳过数
    llm_evaluation_count = 0   # LLM评估数

    async def process_entity_with_hnsw(idx: int, entity: Dict, similar_entities: List[Dict]) -> Dict[str, Any]:
        """只处理LLM评估，HNSW插入在外层完成"""
        async with semaphore:
            entity_id = entity.get("entity_id")

            # 记录处理顺序（用于并查集）
            processing_order.append(entity_id)

            # 直接调用LLM评估（完整ID保留随机ID）
            llm_result = await call_entity_evaluation_llm(
                current_entity=entity,
                similar_entities=similar_entities,
                llm_func=llm_func,
                entity_types_dim1=entity_types_dim1,
                entity_types_dim2=entity_types_dim2
            )

            return {
                "entity": entity,
                "llm_result": llm_result,
                "idx": idx,
                "processing_order_idx": len(processing_order) - 1
            }

    # 顺序处理每个实体：先查询，再插入，再分流处理
    logger.info(f"  开始顺序处理 {len(entities_with_embeddings)} 个实体...")
    llm_coroutines = []  # 使用协程列表

    for idx, entity in enumerate(entities_with_embeddings):
        entity_id = entity.get("entity_id")

        # HNSW查询相似实体（只查询已插入的节点）
        similar_entities = []
        top1_similarity = 0.0  # 最高相似度
        if entity.get("embedding") is not None:
            # 查询已插入的节点（避免自引用和循环引用）
            if len(inserted_indices) > 0:
                try:
                    # 查询已插入节点中的相似实体
                    candidates, distances = index.knn_query(entity["embedding"], k=min(topk, len(inserted_indices)))
                    similar_entities = [entities_with_embeddings[int(c)] for c in candidates[0]
                                      if int(c) < len(entities_with_embeddings) and int(c) in inserted_indices]

                    # 计算相似度分数（转换为相似度）
                    if len(distances[0]) > 0:
                        top1_similarity = 1.0 - distances[0][0]  # 最高相似度

                    logger.debug(f"  实体 {entity_id}: 找到 {len(similar_entities)} 个已插入的相似实体, 最高相似度: {top1_similarity:.3f}")
                except Exception as e:
                    logger.warning(f"实体 {entity_id} HNSW查询失败: {e}")

        # 题目实体特殊处理：题目ID完全匹配优先合并
        is_problem_type = "题目" in entity.get("entity_type_dim2", "")
        if is_problem_type and similar_entities:
            current_problem_id = extract_problem_id(entity_id)
            if current_problem_id:
                same_problem_entities = [
                    e for e in similar_entities
                    if extract_problem_id(e.get("entity_id", "")) == current_problem_id
                ]
                if same_problem_entities:
                    best_match_entity = same_problem_entities[0]
                    best_match_entity_id = best_match_entity.get("entity_id")
                    entities_map[entity_id] = best_match_entity_id
                    auto_merged_count += 1
                    logger.debug(f"  ✓ 题目ID匹配合并: {entity_id} -> {best_match_entity_id} (题目ID: {current_problem_id})")
            continue

        # 立即插入当前实体到HNSW（在查询之后，分流处理之前）
        if entity.get("embedding") is not None:
            index.add_items(entity["embedding"], idx)
            inserted_indices.append(idx)
            logger.debug(f"  实体 {entity_id} 已插入HNSW，索引位置: {idx}")

        # ==================== 根据实体类型分流处理 ====================
        is_problem_entity = entity.get("is_problem_extracted", False)

        if is_problem_entity:
            # 题目实体：直接使用相似度阈值判断
            if similar_entities and top1_similarity >= problem_similarity_threshold:
                # 高相似度：直接合并
                best_match_entity = similar_entities[0]
                best_match_entity_id = best_match_entity.get("entity_id")
                entities_map[entity_id] = best_match_entity_id
                auto_merged_count += 1
                logger.debug(f"  ✓ 题目实体自动合并: {entity_id} -> {best_match_entity_id} (相似度: {top1_similarity:.3f})")
            else:
                # 低相似度：直接跳过
                auto_skipped_count += 1
                logger.debug(f"  ○ 题目实体自动跳过: {entity_id} (相似度: {top1_similarity:.3f})")
        else:
            # 普通实体：三级分流处理
            if similar_entities and top1_similarity >= high_similarity_threshold:
                # 高相似度：直接合并
                best_match_entity = similar_entities[0]
                best_match_entity_id = best_match_entity.get("entity_id")
                entities_map[entity_id] = best_match_entity_id
                auto_merged_count += 1
                logger.debug(f"  ✓ 普通实体自动合并: {entity_id} -> {best_match_entity_id} (相似度: {top1_similarity:.3f})")

            elif top1_similarity <= low_similarity_threshold:
                # 低相似度：直接跳过
                auto_skipped_count += 1
                logger.debug(f"  ○ 普通实体自动跳过: {entity_id} (相似度: {top1_similarity:.3f})")

            else:
                # 中等相似度：调用LLM评估
                coroutine = process_entity_with_hnsw(idx, entity, similar_entities)
                llm_coroutines.append(coroutine)
                llm_evaluation_count += 1

    # 等待所有LLM协程完成
    logger.info(f"  启动 {len(llm_coroutines)} 个并行LLM协程...")
    llm_results = await asyncio.gather(*llm_coroutines, return_exceptions=True)

    # ==================== 阶段2：应用LLM结果并构建并查集 ====================
    logger.info(f"  阶段2：应用LLM结果并构建并查集...")

    # 按处理顺序应用LLM结果（保持并查集的顺序性）
    for result in llm_results:
        if isinstance(result, Exception):
            logger.error(f"LLM任务执行失败: {result}")
            continue

        entity = result["entity"]
        llm_result = result["llm_result"]
        entity_id = entity.get("entity_id")

        # 处理LLM结果
        if llm_result.get("should_delete"):
            deleted_entities.add(entity_id)
            logger.info(f"  删除低质量实体: {entity_id}")
            continue

        # 更新并查集（如果需要合并）
        if llm_result.get("should_merge") and llm_result.get("merge_target"):
            merge_target = llm_result["merge_target"]

            # 调试日志：检查LLM合并指令
            logger.info(f"  LLM指令: {entity_id} 应该合并到 {merge_target}")

            # 检查是否自引用
            if merge_target == entity_id:
                logger.debug(f"  跳过自引用合并: {entity_id} -> {merge_target}")
            # 确保目标不是已删除的实体
            elif merge_target in entities_dict and merge_target not in deleted_entities:
                entities_map[entity_id] = merge_target
                logger.info(f"  ✓ 合并实体: {entity_id} -> {merge_target}")
            else:
                logger.warning(f"  合并目标 {merge_target} 已删除或不存在，取消合并")
        else:
            # 调试日志：检查为什么没有合并指令
            if not llm_result.get("should_merge"):
                logger.debug(f"  {entity_id}: LLM未给出合并指令")
            elif not llm_result.get("merge_target"):
                logger.debug(f"  {entity_id}: LLM未指定合并目标")

    logger.info(f"  阶段2完成：处理了 {len(llm_results)} 个LLM结果")

    # 性能统计
    successful_llm = sum(1 for r in llm_results if not isinstance(r, Exception))
    failed_llm = len(llm_results) - successful_llm
    logger.info(f"  LLM并行处理统计: 成功 {successful_llm}, 失败 {failed_llm}")
    logger.info(f"  分块 '{chunk_name}' HNSW预加载 + LLM并行处理完成")

    # 路径压缩（迭代版本，避免递归深度超限）
    def find_root(entity_id, visited=None):
        """迭代查找根节点，避免递归深度超限"""
        if visited is None:
            visited = set()

        if entity_id in deleted_entities:
            return None

        # 检测循环引用（路径长度大于1的真正循环）
        if entity_id in visited:
            logger.warning(f"检测到循环引用: {entity_id}, 停止查找")
            return entity_id  # 返回自身打破循环

        visited.add(entity_id)

        current = entity_id
        path = []

        # 迭代查找根节点
        while current is not None and current not in deleted_entities and entities_map.get(current, current) != current:
            path.append(current)

            # 防止无限循环（提高阈值，因为合并链可能较长）
            if len(path) > 10000:  # 提高到10000
                logger.warning(f"检测到过长路径: {entity_id} (路径长度: {len(path)}), 停止查找")
                break

            current = entities_map.get(current, current)

        # 如果循环退出时 current 指向自己，说明找到了根节点
        # 如果 current 为 None 或被删除，说明该路径无效
        root = current if current is not None and current not in deleted_entities else None

        # 路径压缩：直接将路径上的所有节点指向根节点
        for node in path:
            entities_map[node] = root if root is not None else node

        return root

    logger.info(f"  开始路径压缩，处理 {len(entities_map)} 个实体...")
    self_ref_count = 0
    for entity_id in list(entities_map.keys()):
        if entity_id not in deleted_entities:
            current_mapping = entities_map.get(entity_id, entity_id)
            root = find_root(entity_id)
            entities_map[entity_id] = root if root is not None else entity_id

            # 统计自引用情况
            if root == entity_id and current_mapping == entity_id:
                self_ref_count += 1

    logger.info(f"  路径压缩完成，发现 {self_ref_count} 个独立实体（根节点自引用）")
    if self_ref_count > 0:
        logger.info(f"  这些实体是独立的（没有合并到其他实体），这是正常状态")

    # 构建合并组（排除已删除的实体）
    merge_groups = {}
    for entity_id, root_id in entities_map.items():
        if entity_id in deleted_entities or root_id is None:
            continue
        if root_id not in merge_groups:
            merge_groups[root_id] = []
        merge_groups[root_id].append(entity_id)

    logger.info(f"分块 '{chunk_name}' LLM评估完成:")
    logger.info(f"  - 原始实体数: {len(entities)}")
    logger.info(f"  - 删除实体数: {len(deleted_entities)}")
    logger.info(f"  - 合并组数: {len(merge_groups)}")

    # ==================== 阶段3：智能合并实体组 ====================
    logger.info(f"  阶段3：智能合并实体组...")

    def count_description_parts(entity_id: str) -> int:
        """计算实体描述中描述片段的数量（分隔符数量+1）"""
        if entity_id not in entities_dict:
            return 0
        description = entities_dict[entity_id].get('description', '')
        if not description:
            return 0
        # 描述片段数 = 分隔符数量 + 1
        separator_count = description.count('<SEP>')
        return separator_count + 1

    # 分类处理不同类型的合并组
    single_entity_groups = []      # 单个实体的组
    small_merge_groups = []        # 小合并组（<=阈值）
    large_merge_groups = []        # 大合并组（>阈值）

    for root_id, group in merge_groups.items():
        if len(group) == 1:
            single_entity_groups.append((root_id, group))
        else:
            # 计算组内所有实体描述片段的总数
            total_parts = sum(count_description_parts(eid) for eid in group)
            # 使用描述片段总数作为阈值判断（避免增量更新导致的描述过度拼接）
            if total_parts <= small_merge_threshold:
                small_merge_groups.append((root_id, group))
            else:
                large_merge_groups.append((root_id, group))

    # 性能统计
    direct_merge_count = 0         # 直接合并组数
    direct_merge_entities = 0      # 直接合并涉及实体数
    llm_merge_count = 0            # LLM合并组数
    llm_merge_entities = 0         # LLM合并涉及实体数

    merged_entities = []

    # 处理单实体组（无需处理，直接保留）
    for root_id, group in single_entity_groups:
        entity = entities_dict[group[0]].copy()
        merged_entities.append(entity)

    # 处理小合并组（直接拼接，无需LLM）
    if small_merge_groups:
        logger.info(f"  直接合并 {len(small_merge_groups)} 个小合并组（<= {small_merge_threshold} 个描述片段）...")

        GRAPH_FIELD_SEP = "<SEP>"

        for root_id, group in small_merge_groups:
            # 准备组内实体数据
            group_entities = []
            for eid in group:
                entity = entities_dict[eid].copy()
                group_entities.append(entity)

            # 直接合并逻辑
            main_entity = group_entities[0].copy()  # 取第一个作为主实体

            # 合并实体类型（取并集）
            all_dim1_types = []
            all_dim2_types = []
            for entity in group_entities:
                dim1 = entity.get('entity_type_dim1', '')
                dim2 = entity.get('entity_type_dim2', '')
                if dim1:
                    all_dim1_types.extend([t.strip() for t in dim1.replace('，', ',').split(',') if t.strip()])
                if dim2:
                    all_dim2_types.extend([t.strip() for t in dim2.replace('，', ',').split(',') if t.strip()])

            # 去重并合并类型
            main_entity['entity_type_dim1'] = ','.join(list(set(all_dim1_types)))
            main_entity['entity_type_dim2'] = ','.join(list(set(all_dim2_types)))

            # 拼接描述
            descriptions = [e.get('description', '') for e in group_entities if e.get('description')]
            if descriptions:
                main_entity['description'] = GRAPH_FIELD_SEP.join(descriptions)

            # 记录合并信息
            main_entity['merged_from'] = group
            main_entity['merge_count'] = len(group)
            main_entity['merge_timestamp'] = int(time.time())
            main_entity['merge_method'] = 'direct_concat'  # 标记为直接拼接合并

            merged_entities.append(main_entity)
            direct_merge_count += 1
            direct_merge_entities += len(group)
            logger.debug(f"  ✓ 直接合并: {group} -> {main_entity.get('entity_id', 'unknown')} ({len(group)}个实体)")

    # 处理大合并组（调用LLM）
    if large_merge_groups:
        logger.info(f"  LLM合并 {len(large_merge_groups)} 个大合并组（> {small_merge_threshold} 个描述片段）...")
        merge_coroutines = []

        for root_id, group in large_merge_groups:
            # 准备组内实体数据
            group_entities = []
            for eid in group:
                entity = entities_dict[eid].copy()
                group_entities.append(entity)

            # 创建LLM合并协程
            async def merge_group(group, group_entities):
                """异步合并单个实体组"""
                merge_result = await call_entity_group_merge_llm(
                    entities=group_entities,
                    llm_func=llm_func,
                    entity_types_dim1=entity_types_dim1,
                    entity_types_dim2=entity_types_dim2
                )

                # 构建合并后的实体
                main_entity = group_entities[0].copy()
                main_entity["entity_id"] = merge_result["final_name"]
                main_entity["entity_type_dim1"] = merge_result["final_type_dim1"]
                main_entity["entity_type_dim2"] = merge_result["final_type_dim2"]
                main_entity["description"] = merge_result["final_description"]
                main_entity["merged_from"] = group
                main_entity["merge_count"] = len(group)
                main_entity["merge_timestamp"] = int(time.time())
                main_entity["merge_method"] = 'llm_merge'  # 标记为LLM合并

                logger.info(f"  ✓ LLM合并: {group} -> {merge_result['final_name']}")
                return main_entity

            coroutine = merge_group(group, group_entities)
            merge_coroutines.append(coroutine)

        # 并行等待所有LLM合并完成
        logger.info(f"  启动 {len(merge_coroutines)} 个并行实体组合并协程...")
        merge_results = await asyncio.gather(*merge_coroutines, return_exceptions=True)

        # 收集合并结果
        for result in merge_results:
            if isinstance(result, Exception):
                logger.error(f"实体组合并失败: {result}")
                continue
            merged_entities.append(result)
            llm_merge_count += 1
            llm_merge_entities += len(result.get('merged_from', []))

        logger.info(f"  实体组合并完成，成功处理 {len(merge_results)} 个大实体组合并")
    else:
        logger.info(f"  没有需要LLM合并的大实体组")

    logger.info(f"分块 '{chunk_name}' 处理完成，最终实体数: {len(merged_entities)}")

    # 构建完整的实体ID映射：原始ID -> 最终实体ID
    # 注意：合并后的实体有新的entity_id（final_name），需要正确映射
    entity_id_mapping = {}

    # 1. 首先标记所有被删除的实体
    for original_id in deleted_entities:
        entity_id_mapping[original_id] = None

    # 2. 从合并后的实体中构建映射
    for merged_entity in merged_entities:
        final_id = merged_entity.get("entity_id")
        merged_from = merged_entity.get("merged_from", [])

        if merged_from:
            # 多实体合并：将所有原始ID都映射到新的final_id
            for original_id in merged_from:
                entity_id_mapping[original_id] = final_id
        else:
            # 单实体（未合并）：映射到自身
            entity_id_mapping[final_id] = final_id

    # 3. 补充任何未映射的实体（保持原ID）
    for original_id in entities_dict.keys():
        if original_id not in entity_id_mapping and original_id not in deleted_entities:
            # 查找该实体在并查集中的根节点
            root_id = entities_map.get(original_id, original_id)
            # 如果根节点已经有映射，使用该映射
            if root_id in entity_id_mapping:
                entity_id_mapping[original_id] = entity_id_mapping[root_id]
            else:
                entity_id_mapping[original_id] = original_id

    logger.info(f"构建实体ID映射: {len(entity_id_mapping)} 条映射")

    # 计算性能统计
    total_entities = len(entities)
    total_similarity_decisions = auto_merged_count + auto_skipped_count + llm_evaluation_count
    similarity_api_saved = auto_merged_count + auto_skipped_count
    similarity_api_reduction_rate = similarity_api_saved / total_similarity_decisions if total_similarity_decisions > 0 else 0

    total_merge_groups = len(small_merge_groups) + len(large_merge_groups)
    merge_api_saved = len(small_merge_groups)
    merge_api_reduction_rate = merge_api_saved / total_merge_groups if total_merge_groups > 0 else 0

    total_llm_calls_saved = similarity_api_saved + merge_api_saved
    total_llm_reduction_rate = total_llm_calls_saved / (total_similarity_decisions + total_merge_groups) if (total_similarity_decisions + total_merge_groups) > 0 else 0

    logger.info(f"性能优化统计:")
    logger.info(f"  相似度分流: 自动合并{auto_merged_count}, 自动跳过{auto_skipped_count}, LLM评估{llm_evaluation_count}")
    logger.info(f"  合并组处理: 直接合并{direct_merge_count}组, LLM合并{llm_merge_count}组")
    logger.info(f"  LLM调用节省: 相似度决策节省{similarity_api_saved}, 合并组节省{merge_api_saved}")
    logger.info(f"  总体节省率: {total_llm_reduction_rate:.1%} (从{total_similarity_decisions + total_merge_groups}次降至{total_similarity_decisions - similarity_api_saved + total_merge_groups - merge_api_saved}次)")

    return {
        "chunk_name": chunk_name,
        "entity_count": len(entities),
        "deleted_count": len(deleted_entities),
        "status": "processed",
        "merged_entities": merged_entities,
        "map": entities_map,
        "deleted_entities": deleted_entities,
        "entity_id_mapping": entity_id_mapping,
        # 性能优化统计
        "performance_stats": {
            "auto_merged_count": auto_merged_count,
            "auto_skipped_count": auto_skipped_count,
            "llm_evaluation_count": llm_evaluation_count,
            "direct_merge_count": direct_merge_count,
            "direct_merge_entities": direct_merge_entities,
            "llm_merge_count": llm_merge_count,
            "llm_merge_entities": llm_merge_entities,
            "total_entities": total_entities,
            "total_llm_calls_saved": total_llm_calls_saved,
            "total_llm_reduction_rate": total_llm_reduction_rate,
            "similarity_api_saved": similarity_api_saved,
            "similarity_api_reduction_rate": similarity_api_reduction_rate,
            "merge_api_saved": merge_api_saved,
            "merge_api_reduction_rate": merge_api_reduction_rate
        }
    }


async def process_entity_chunks_with_llm(
    entity_chunks: Dict[str, List[Dict[str, Any]]],
    llm_func: Callable,
    entity_types_dim1: List[str] = None,
    entity_types_dim2: List[str] = None,
    topk: int = None,
    max_concurrent: int = None,
    max_workers: int = None
) -> Dict[str, Dict[str, Any]]:
    """
    使用LLM多线程处理所有实体分块（分块间多线程并行，分块内顺序执行）

    Args:
        entity_chunks: 按dim1类型分块的实体字典
        llm_func: LLM调用函数
        entity_types_dim1: 第一维度类型列表
        entity_types_dim2: 第二维度类型列表
        topk: 检索相似实体的数量
        max_concurrent: 分块内LLM最大并发数
        max_workers: 分块间最大线程数，默认为CPU核心数

    Returns:
        Dict[str, Dict[str, Any]]: 每个分块的处理结果
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if max_workers is None:
        max_workers = min(len(entity_chunks), os.cpu_count() or 4)

    logger.info(f"开始多线程LLM增强处理 {len(entity_chunks)} 个实体分块，使用 {max_workers} 个线程")

    def process_chunk_in_thread(chunk_name: str, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """在线程中运行异步分块处理"""
        try:
            # 在线程中创建新的事件循环运行异步函数
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    process_entity_chunk_with_llm(
                        chunk_name=chunk_name,
                        entities=entities,
                        llm_func=llm_func,
                        entity_types_dim1=entity_types_dim1,
                        entity_types_dim2=entity_types_dim2,
                        topk=topk,
                        max_concurrent=max_concurrent
                    )
                )
                return result
            finally:
                loop.close()
        except Exception as exc:
            logger.error(f"分块 '{chunk_name}' 处理失败: {exc}")
            return {
                "chunk_name": chunk_name,
                "status": "failed",
                "error": str(exc),
                "processed_at": time.time()
            }

    results = {}

    # 使用 ThreadPoolExecutor 并行处理分块
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {}
        for chunk_name, entities in entity_chunks.items():
            future = executor.submit(process_chunk_in_thread, chunk_name, entities)
            future_to_chunk[future] = chunk_name
            logger.info(f"提交分块 '{chunk_name}' 到线程池，包含 {len(entities)} 个实体")

        # 收集处理结果
        for future in as_completed(future_to_chunk):
            chunk_name = future_to_chunk[future]
            try:
                result = future.result()
                results[chunk_name] = result
            except Exception as exc:
                logger.error(f"分块 '{chunk_name}' 处理失败: {exc}")
                results[chunk_name] = {
                    "chunk_name": chunk_name,
                    "status": "failed",
                    "error": str(exc),
                    "processed_at": time.time()
                }

    logger.info(f"多线程LLM增强处理完成，成功处理 {sum(1 for r in results.values() if r.get('status') == 'processed')} 个分块")

    return results


def build_entity_merge_mapping(
    chunk_results: Dict[str, Dict[str, Any]],
    all_entities: List[Dict[str, Any]] = None
) -> Dict[str, Optional[str]]:
    """
    根据实体分块合并结果构建全局实体映射关系

    Args:
        chunk_results: 实体分块处理结果
        all_entities: 所有实体列表（用于补充映射）

    Returns:
        Dict[str, Optional[str]]: 原始实体ID -> 最终实体ID的映射
                                  None表示该实体已被删除
    """
    entity_mapping = {}

    for chunk_name, result in chunk_results.items():
        if result.get("status") != "processed":
            continue

        # 获取当前分块的实体ID映射
        chunk_mapping = result.get("entity_id_mapping", {})
        entity_mapping.update(chunk_mapping)

        # 统计信息
        deleted_count = sum(1 for v in chunk_mapping.values() if v is None)
        renamed_count = sum(1 for k, v in chunk_mapping.items() if v is not None and k != v)
        logger.info(f"分块 '{chunk_name}': {len(chunk_mapping)} 个映射, {deleted_count} 个删除, {renamed_count} 个重命名/合并")

    # 补充映射：建立完整的实体ID映射关系
    if all_entities:
        logger.info("补充完整实体映射...")
        unmapped_count = 0
        merged_count = 0

        for entity in all_entities:
            entity_id = entity.get("entity_id") or entity.get("id") or entity.get("entity_name", "")
            if not entity_id:
                continue

            # 如果实体有 merged_from 字段，说明它是被合并后的结果
            # 需要将被合并的所有实体ID都映射到最终ID
            merged_from = entity.get("merged_from", [])
            if merged_from:
                # 将所有被合并的实体ID都映射到最终实体ID
                for old_id in merged_from:
                    entity_mapping[old_id] = entity_id
                merged_count += 1
                logger.debug(f"合并实体映射: {merged_from} -> {entity_id}")
            elif entity_id not in entity_mapping:
                # 独立实体，没有被合并，保持原ID
                entity_mapping[entity_id] = entity_id
                unmapped_count += 1

        logger.info(f"补充映射: {merged_count} 个合并实体, {unmapped_count} 个独立实体")

    # 统计全局信息
    total_deleted = sum(1 for v in entity_mapping.values() if v is None)
    total_renamed = sum(1 for k, v in entity_mapping.items() if v is not None and k != v)
    logger.info(f"全局实体映射: {len(entity_mapping)} 个实体, {total_deleted} 个删除, {total_renamed} 个重命名/合并")

    return entity_mapping


def chunk_relations_by_merge_groups(relations: List[Dict[str, Any]], entity_mapping: Dict[str, Optional[str]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
    根据实体合并映射对关系进行分块

    Args:
        relations: 原始关系列表
        entity_mapping: 实体映射关系（原始ID -> 最终ID，None表示删除）

    Returns:
        Dict[Tuple[str, str], List[Dict[str, Any]]]: 按合并组分块的关系字典
    """
    relation_chunks = {}
    removed_self_loops = 0
    removed_deleted_entity = 0
    processed_relations = 0

    for relation in relations:
        src_id = relation.get("src_id")
        tgt_id = relation.get("tgt_id")

        if not src_id or not tgt_id:
            continue

        # 获取实体映射后的最终ID
        src_final = entity_mapping.get(src_id, src_id)
        tgt_final = entity_mapping.get(tgt_id, tgt_id)

        # 调试日志：如果ID不在映射中，记录下来
        if src_id not in entity_mapping:
            logger.debug(f"源ID不在映射中: {src_id}")
        if tgt_id not in entity_mapping:
            logger.debug(f"目标ID不在映射中: {tgt_id}")

        # 如果任一端实体被删除（映射为None），删除该关系
        if src_final is None or tgt_final is None:
            removed_deleted_entity += 1
            logger.debug(f"删除关系（端点实体已删除）: {src_id} -> {tgt_id}")
            continue

        # 如果映射后的源和目标是同一个实体，说明是自环，删除
        if src_final == tgt_final:
            removed_self_loops += 1
            logger.debug(f"删除自环关系: {src_id} -> {tgt_id} (映射后: {src_final})")
            continue

        # 使用映射后的最终ID作为分块键
        chunk_key = (src_final, tgt_final)

        if chunk_key not in relation_chunks:
            relation_chunks[chunk_key] = []

        # 创建关系副本，并更新为映射后的ID
        processed_relation = relation.copy()
        processed_relation["src_id"] = src_final
        processed_relation["tgt_id"] = tgt_final
        processed_relation["original_src_id"] = src_id  # 保留原始ID用于调试
        processed_relation["original_tgt_id"] = tgt_id  # 保留原始ID用于调试

        relation_chunks[chunk_key].append(processed_relation)
        processed_relations += 1

    logger.info(f"关系分块完成:")
    logger.info(f"  - 原始关系数: {len(relations)}")
    logger.info(f"  - 删除（端点实体已删除）: {removed_deleted_entity}")
    logger.info(f"  - 删除（自环）: {removed_self_loops}")
    logger.info(f"  - 处理关系数: {processed_relations}")
    logger.info(f"  - 关系分块数: {len(relation_chunks)}")

    return relation_chunks


async def process_relation_chunks_async(relation_chunks: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    异步版本的关系分块处理，支持LLM生成描述
    
    Args:
        relation_chunks: 按合并组分块的关系字典
        
    Returns:
        Dict[Tuple[str, str], Dict[str, Any]]: 每个分块的处理结果
    """
    logger.info(f"开始异步处理 {len(relation_chunks)} 个关系分块")
    
    results = {}
    
    # 并发处理所有分块
    tasks = []
    for chunk_key, relations in relation_chunks.items():
        task = process_single_relation_chunk_async(chunk_key, relations)
        tasks.append((chunk_key, task))
    
    # 等待所有任务完成
    for chunk_key, task in tasks:
        try:
            result = await task
            results[chunk_key] = result
        except Exception as exc:
            logger.error(f"关系分块 '{chunk_key[0]} -> {chunk_key[1]}' 处理失败: {exc}")
            results[chunk_key] = {
                "chunk_key": chunk_key,
                "status": "failed",
                "error": str(exc),
                "processed_at": time.time()
            }
    
    logger.info(f"异步关系处理完成，成功处理 {sum(1 for r in results.values() if r.get('status') == 'processed')} 个分块")
    
    return results


async def process_single_relation_chunk_async(chunk_key: Tuple[str, str], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """异步处理单个关系分块"""
    src_root, tgt_root = chunk_key
    GRAPH_FIELD_SEP = "<SEP>"
    FORCE_LLM_THRESHOLD = 4

    # 只有1个关系，不需要合并
    if len(relations) == 1:
        return {"chunk_key": chunk_key, "relation_count": 1, "status": "processed", "result": relations[0]}

    # 收集所有描述、关键词、权重等
    descriptions, keywords_set, total_weight = [], set(), 0.0
    file_paths, source_chunk_ids = [], []

    for rel in relations:
        if rel.get("description"):
            descriptions.append(rel["description"])
        if rel.get("keywords"):
            keywords_set.update(kw.strip() for kw in rel["keywords"].split(",") if kw.strip())
        total_weight += float(rel.get("weight", 1.0))
        if rel.get("file_path"):
            file_paths.append(rel["file_path"])
        if rel.get("source_chunk_id"):
            source_chunk_ids.append(rel["source_chunk_id"])

    # 去重描述
    unique_descriptions = list(dict.fromkeys(descriptions))

    # 合并描述：少量直接拼接，大量使用LLM生成
    if len(unique_descriptions) <= FORCE_LLM_THRESHOLD:
        final_description = GRAPH_FIELD_SEP.join(unique_descriptions)
    else:
        # 使用LLM摘要服务生成描述
        try:
            llm_service = LLMSummaryService()
            relation_name = f"{src_root} -> {tgt_root}"
            final_description = await llm_service.summarize_descriptions(
                description_type="Relation",
                description_name=relation_name,
                description_list=unique_descriptions,
                separator=GRAPH_FIELD_SEP,
                summary_length="One Paragraphs",
                language="zh-CN"
            )
            logger.info(final_description)
        except Exception as e:
            logger.warning(f"LLM描述生成失败: {e}, 使用原始描述拼接")
            final_description = GRAPH_FIELD_SEP.join(unique_descriptions)

    # 构建合并后的关系
    merged_relation = {
        "src_id": src_root,
        "tgt_id": tgt_root,
        "description": final_description,
        "keywords": ",".join(sorted(keywords_set)),
        "weight": total_weight,
        "file_path": GRAPH_FIELD_SEP.join(list(dict.fromkeys(file_paths))),
        "source_chunk_id": GRAPH_FIELD_SEP.join(list(dict.fromkeys(source_chunk_ids))),
    }

    logger.info(f"  → 关系合并完成: {src_root} -> {tgt_root}")
    logger.info(f"    关系数: {len(relations)}, 权重: {total_weight}")
    logger.info(f"    关键词: {len(keywords_set)} 个")
    logger.info(f"    嵌入向量: 不为关系生成嵌入向量（通过端点实体计算相似度）")

    return {"chunk_key": chunk_key, "relation_count": len(relations), "status": "processed", "result": merged_relation}


async def process_relation_chunks_multithreaded(relation_chunks: Dict[Tuple[str, str], List[Dict[str, Any]]], max_workers: int = None, llm_func=None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    多线程处理关系分块
    
    Args:
        relation_chunks: 按合并组分块的关系字典
        max_workers: 最大线程数，默认为CPU核心数
        llm_func: LLM调用函数，用于生成描述摘要
        
    Returns:
        Dict[Tuple[str, str], Dict[str, Any]]: 每个分块的处理结果
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    if max_workers is None:
        max_workers = min(len(relation_chunks), os.cpu_count() or 4)
    
    logger.info(f"开始多线程处理 {len(relation_chunks)} 个关系分块，使用 {max_workers} 个线程")
    
    results = {}
    
    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有分块处理任务
        future_to_chunk = {}
        for chunk_key, relations in relation_chunks.items():
            # 传递llm_func到线程池任务
            future = executor.submit(process_relation_chunk_sync, chunk_key, relations, llm_func)
            future_to_chunk[future] = chunk_key
            #logger.info(f"提交关系分块 '{chunk_key[0]} -> {chunk_key[1]}' 到线程池，包含 {len(relations)} 个关系")
        
        # 收集处理结果
        for future in as_completed(future_to_chunk):
            chunk_key = future_to_chunk[future]
            try:
                result = future.result()
                results[chunk_key] = result
                #logger.info(f"关系分块 '{chunk_key[0]} -> {chunk_key[1]}' 处理完成")
            except Exception as exc:
                logger.error(f"关系分块 '{chunk_key[0]} -> {chunk_key[1]}' 处理失败: {exc}")
                results[chunk_key] = {
                    "chunk_key": chunk_key,
                    "status": "failed",
                    "error": str(exc),
                    "processed_at": time.time()
                }
    
    logger.info(f"多线程关系处理完成，成功处理 {sum(1 for r in results.values() if r.get('status') == 'processed')} 个分块")
    
    return results


def process_relation_chunk_sync(chunk_key: Tuple[str, str], relations: List[Dict[str, Any]], llm_func=None) -> Dict[str, Any]:
    """
    同步版本的process_relation_chunk，用于线程池调用

    Args:
        chunk_key: 分块键 (src_root_id, tgt_root_id)
        relations: 该分块中的关系列表
        llm_func: LLM调用函数，用于生成描述摘要

    Returns:
        Dict[str, Any]: 处理结果，包含合并后的relation
    """
    src_root, tgt_root = chunk_key
    GRAPH_FIELD_SEP = "<SEP>"
    FORCE_LLM_THRESHOLD = 4

    # 只有1个关系，不需要合并
    if len(relations) == 1:
        return {"chunk_key": chunk_key, "relation_count": 1, "status": "processed", "result": relations[0]}

    # 收集所有描述、关键词、权重等
    descriptions, keywords_set, total_weight = [], set(), 0.0
    file_paths, source_chunk_ids = [], []

    for rel in relations:
        if rel.get("description"):
            descriptions.append(rel["description"])
        if rel.get("keywords"):
            keywords_set.update(kw.strip() for kw in rel["keywords"].split(",") if kw.strip())
        total_weight += float(rel.get("weight", 1.0))
        if rel.get("file_path"):
            file_paths.append(rel["file_path"])
        if rel.get("source_chunk_id"):
            source_chunk_ids.append(rel["source_chunk_id"])

    # 去重描述
    unique_descriptions = list(dict.fromkeys(descriptions))

    # 合并描述：少量直接拼接，大量使用LLM生成
    if len(unique_descriptions) <= FORCE_LLM_THRESHOLD:
        final_description = GRAPH_FIELD_SEP.join(unique_descriptions)
    else:
        # 同步版本：直接拼接，不调用LLM避免timeout
        logger.info(f"跳过LLM调用，直接拼接 {len(unique_descriptions)} 个描述")
        final_description = GRAPH_FIELD_SEP.join(unique_descriptions)

    # 构建合并后的关系
    merged_relation = {
        "src_id": src_root,
        "tgt_id": tgt_root,
        "description": final_description,
        "keywords": ",".join(sorted(keywords_set)),
        "weight": total_weight,
        "file_path": GRAPH_FIELD_SEP.join(list(dict.fromkeys(file_paths))),
        "source_chunk_id": GRAPH_FIELD_SEP.join(list(dict.fromkeys(source_chunk_ids))),
    }

    logger.info(f"  → 关系合并完成: {src_root} -> {tgt_root}")
    logger.info(f"    关系数: {len(relations)}, 权重: {total_weight}")
    logger.info(f"    关键词: {len(keywords_set)} 个")

    return {"chunk_key": chunk_key, "relation_count": len(relations), "status": "processed", "result": merged_relation}

# 使用示例
async def main():
    """
    主函数 - 从新架构输出加载数据并进行实体合并

    使用方式：
    1. 先运行 new_test.py 生成数据到 ./new_rag_storage/
    2. 再运行本文件进行实体合并
    """
    from dotenv import load_dotenv
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed

    load_dotenv()

    try:
        # 从环境变量获取输入目录
        input_dir = os.getenv("ENTITY_MERGE_INPUT_DIR", "./new_rag_storage")
        entities_file = os.path.join(input_dir, "entities.json")
        relations_file = os.path.join(input_dir, "relations.json")

        # 检查文件是否存在
        if not os.path.exists(entities_file):
            logger.error(f"实体文件不存在: {entities_file}")
            logger.info("请先运行 new_test.py 生成数据")
            return

        # 创建数据加载器（使用环境变量中的路径）
        merged_data_dir = os.getenv("ENTITY_MERGE_OUTPUT_DIR", "./merged_data")
        loader = EntityDataLoader(output_dir=merged_data_dir)

        # 从 JSON 文件加载数据
        entities, relations = loader.load_from_json(
            entities_file=entities_file,
            relations_file=relations_file
        )

        if not entities:
            logger.warning("未加载到任何实体数据")
            return

        logger.info(f"加载到 {len(entities)} 个实体, {len(relations)} 个关系")

        # ==================== 生成嵌入向量 ====================
        logger.info("=" * 60)
        logger.info("生成实体嵌入向量")
        logger.info("=" * 60)

        # 配置嵌入函数
        async def embedding_func(texts: List[str]) -> List[List[float]]:
            return await openai_embed(
                texts,
                model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
                base_url=os.getenv("EMBEDDING_BINDING_HOST", "https://api.siliconflow.cn/v1"),
                api_key=os.getenv("EMBEDDING_BINDING_API_KEY"),
            )

        # 批量生成嵌入向量
        embedding_count = await loader.generate_embeddings(
            embedding_func=embedding_func,
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
            max_concurrent=int(os.getenv("EMBEDDING_MAX_CONCURRENT", "20"))
        )

        logger.info(f"成功为 {embedding_count} 个实体生成嵌入向量")

        # 更新 entities 引用（因为 loader.entities 已经被修改）
        entities = loader.entities

        # 配置LLM函数
        async def llm_func(prompt: str) -> str:
            return await openai_complete_if_cache(
                os.getenv("LLM_MODEL", "deepseek-chat"),
                prompt,
                api_key=os.getenv("LLM_BINDING_API_KEY"),
                base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
            )

        # ==================== 第一步：预处理UNKNOWN类型实体 ====================
        logger.info("=" * 60)
        logger.info("第一步：预处理UNKNOWN类型实体")
        logger.info("=" * 60)

        entities, unknown_deleted = await reclassify_unknown_entities(
            entities=entities,
            llm_func=llm_func,
            entity_types_dim1=DEFAULT_ENTITY_TYPES_DIM1,
            entity_types_dim2=DEFAULT_ENTITY_TYPES_DIM2,
            max_concurrent=DEFAULT_ENTITY_MERGE_MAX_ASYNC
        )

        logger.info(f"预处理完成，剩余 {len(entities)} 个实体")

        # ==================== 第二步：按dim1类型分块 ====================
        logger.info("=" * 60)
        logger.info("第二步：按类型分块并进行双策略实体合并")
        logger.info("=" * 60)

        entity_chunks = chunk_entities_by_dim1_type(entities)

        # 使用双策略处理实体分块（普通实体用LLM，题目实体用快速策略）
        logger.info("开始双策略处理实体分块...")
        dual_config = DualStrategyMergeConfig()
        chunk_results = await process_entity_chunks_with_dual_strategy(
            entity_chunks=entity_chunks,
            llm_func=llm_func,
            entity_types_dim1=DEFAULT_ENTITY_TYPES_DIM1,
            entity_types_dim2=DEFAULT_ENTITY_TYPES_DIM2,
            topk=DEFAULT_ENTITY_MERGE_TOPK,
            max_concurrent=DEFAULT_ENTITY_MERGE_MAX_ASYNC,
            dual_config=dual_config
        )

        # 收集合并后的实体列表
        merged_entities = collect_merged_entities_from_results(chunk_results)

        # 打印处理结果摘要
        print("\n" + "="*60)
        print("双策略实体合并结果摘要")
        print("="*60)

        total_entities = 0
        total_deleted = 0
        total_normal = 0
        total_problem = 0
        successful_chunks = 0
        failed_chunks = 0

        for chunk_name, result in chunk_results.items():
            entity_count = result.get("entity_count", 0)
            deleted_count = result.get("deleted_count", 0)
            normal_count = result.get("normal_entity_count", 0)
            problem_count = result.get("problem_entity_count", 0)
            status = result.get("status", "unknown")
            total_entities += entity_count
            total_deleted += deleted_count
            total_normal += normal_count
            total_problem += problem_count

            if status == "processed":
                successful_chunks += 1
                merged_count = len(result.get("merged_entities", []))
                print(f"  分块 '{chunk_name}': 普通 {normal_count}, 题目 {problem_count}, 删除 {deleted_count}, 合并后 {merged_count}")
            else:
                failed_chunks += 1
                print(f"  分块 '{chunk_name}': {entity_count} 个实体 - {status}")

        print(f"\n总计: {len(entity_chunks)} 个分块, {total_entities} 个原始实体")
        print(f"  - 普通实体: {total_normal} (使用LLM高质量策略)")
        print(f"  - 题目实体: {total_problem} (使用快速相似度策略)")
        print(f"删除低质量实体: {total_deleted}")
        print(f"最终合并实体: {len(merged_entities)}")
        print(f"成功: {successful_chunks} 个分块, 失败: {failed_chunks} 个分块")
        print("="*60)

        # ==================== 关系合并流程 ====================
        if relations:
            logger.info("开始关系合并流程...")

            # 构建实体映射关系（传入合并后的实体以正确处理merged_from字段）
            logger.info("构建实体映射关系...")
            entity_mapping = build_entity_merge_mapping(chunk_results, all_entities=merged_entities)

            # 关系分块
            logger.info("开始关系分块...")
            relation_chunks = chunk_relations_by_merge_groups(relations, entity_mapping)

            # 异步处理关系分块
            logger.info("开始异步处理关系分块...")
            relation_chunk_results = await process_relation_chunks_async(relation_chunks)

            # 收集合并后的关系列表
            merged_relations = collect_merged_relations_from_results(relation_chunk_results)

            logger.info(f"关系合并完成: {len(merged_relations)} 个关系")
        else:
            merged_relations = []
            entity_mapping = {}
            logger.info("无关系数据需要处理")

        # ==================== 保存合并后的数据到文件 ====================
        logger.info("保存合并后的数据到文件...")

        # 更新 loader 的数据
        loader.entities = merged_entities
        loader.relations = merged_relations

        # 保存到文件
        loader.save_data()

        # 打印摘要
        loader.print_summary()

        # ==================== 存入Neo4j ====================
        logger.info("=" * 60)
        logger.info("存入Neo4j数据库")
        logger.info("=" * 60)

        # 是否清空现有数据（可通过环境变量配置）
        clear_neo4j = os.getenv("CLEAR_NEO4J", "true").lower() == "true"

        neo4j_stats = await save_to_neo4j(
            merged_entities=merged_entities,
            merged_relations=merged_relations,
            entity_mapping=entity_mapping,
            clear_existing=clear_neo4j
        )

        print("\n" + "="*60)
        print("实体合并完成!")
        print("="*60)
        print(f"输入: {entities_file}")
        print(f"输出: {loader.output_dir}/")
        print(f"合并后实体: {len(merged_entities)}")
        print(f"合并后关系: {len(merged_relations)}")
        print(f"Neo4j实体: {neo4j_stats['entities_upserted']}")
        print(f"Neo4j关系: {neo4j_stats['relations_upserted']}")
        print("="*60)

    except Exception as e:
        logger.error(f"主函数执行失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def clean_relation_keywords(keywords_str: str, description: str) -> tuple[list[str], str]:
    """清洗关系keyword，提取标准关系类型并处理污染内容"""
    VALID_RELATION_TYPES = {
        "IS_A", "PART_OF", "BASED_ON", "APPLIES_TO", "EVALUATES",
        "EXPLAINS", "PRACTICED_BY", "COMPARES_WITH", "LEADS_TO",
        "OPTIMIZES", "TRANSFORMS_TO"
    }

    if not keywords_str:
        return [], description

    # 尝试匹配标准关系类型
    cleaned_keywords = []
    original_keywords = [k.strip() for k in keywords_str.replace('，', ',').split(',') if k.strip()]

    for kw in original_keywords:
        if kw in VALID_RELATION_TYPES:
            cleaned_keywords.append(kw)
        else:
            # 查找kw中的标准类型前缀
            found = False
            for valid_type in VALID_RELATION_TYPES:
                if kw.startswith(valid_type + ','):
                    cleaned_keywords.append(valid_type)
                    # 剩余部分加入description
                    remaining = kw[len(valid_type) + 1:].strip()
                    if remaining:
                        description = f"{description} [额外关键词: {remaining}]" if description else f"[额外关键词: {remaining}]"
                    found = True
                    break

            if not found and kw:
                # 未找到标准类型，保留原keyword并在description中标注
                description = f"{description} [未识别关键词: {kw}]" if description else f"[未识别关键词: {kw}]"

    return cleaned_keywords, description


async def save_to_neo4j(
    merged_entities: List[Dict[str, Any]],
    merged_relations: List[Dict[str, Any]],
    entity_mapping: Dict[str, Optional[str]] = None,
    clear_existing: bool = True
) -> Dict[str, Any]:
    """
    将合并后的实体和关系数据存入Neo4j（独立函数，不依赖LightRAG）

    Args:
        merged_entities: 合并后的实体列表
        merged_relations: 合并后的关系列表
        entity_mapping: 实体ID映射表（原始ID -> 最终ID，None表示删除）
        clear_existing: 是否在写入前清空现有数据（谨慎使用）

    Returns:
        Dict[str, Any]: 操作结果统计
    """
    from py2neo import Graph, Node, Relationship
    from dotenv import load_dotenv

    logger.info("=== 开始执行 save_to_neo4j ===")

    stats = {
        "entities_upserted": 0,
        "entities_failed": 0,
        "relations_upserted": 0,
        "relations_failed": 0,
        "relations_self_loops_removed": 0,
        "entities_cleared": 0,
        "relations_cleared": 0,
        "errors": []
    }

    logger.info(f"开始将合并数据存入Neo4j: {len(merged_entities)} 个实体, {len(merged_relations)} 个关系")

    # 调试：检查前几个关系的ID格式
    if merged_relations:
        logger.info("前5个关系的ID示例:")
        for i, rel in enumerate(merged_relations[:5]):
            logger.info(f"  {i+1}. {rel.get('src_id')} -> {rel.get('tgt_id')}")

    # ==================== 连接Neo4j ====================
    load_dotenv(dotenv_path=".env", override=False)
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))

    # ==================== 清空Neo4j数据（如果需要） ====================
    if clear_existing:
        logger.warning("清空模式：将先删除所有Neo4j数据，再重新存入合并数据")
        try:
            # 使用显式事务确保清空操作可靠执行
            tx = graph.begin()
            result = tx.run("MATCH (n) DETACH DELETE n")
            graph.commit(tx)

            # 验证清空操作结果
            stats["entities_cleared"] = len(merged_entities)
            stats["relations_cleared"] = len(merged_relations)
            logger.info("已清空所有Neo4j数据")

            # 再次验证：检查是否还有剩余节点
            remaining_count = graph.run("MATCH (n) RETURN count(n) as count").evaluate()
            logger.info(f"清空后剩余节点数: {remaining_count}")

        except Exception as e:
            logger.error(f"清空Neo4j数据失败: {e}")
            stats["errors"].append(f"清空Neo4j数据失败: {str(e)}")

    # ==================== 插入实体数据 ====================
    logger.info("开始插入实体数据...")

    # 设置实体批处理大小（减小批次避免超时）
    ENTITY_BATCH_SIZE = 500  # 从1000减少到500
    entity_batches = [merged_entities[i:i + ENTITY_BATCH_SIZE]
                      for i in range(0, len(merged_entities), ENTITY_BATCH_SIZE)]

    logger.info(f"实体总数: {len(merged_entities)}, 批次数: {len(entity_batches)}, 每批大小: {ENTITY_BATCH_SIZE}")

    # 分批插入实体
    for batch_idx, entity_batch in enumerate(entity_batches):
        tx = graph.begin()
        try:
            for entity in entity_batch:
                try:
                    entity_id = entity.get("entity_id") or entity.get("id") or entity.get("entity_name", "")
                    if not entity_id:
                        stats["entities_failed"] += 1
                        continue

                    # 使用py2neo创建节点
                    node_props = {
                        "entity_id": entity_id,
                        "description": entity.get("description", ""),
                        "source_id": entity.get("source_id", ""),
                        "file_path": entity.get("file_path", ""),
                        "created_at": entity.get("created_at", int(time.time()))
                    }

                    # 添加嵌入向量（如果存在）
                    if entity.get("embedding") is not None:
                        node_props["embedding"] = entity["embedding"]

                    # 解析entity_type_dim1（支持中英文逗号）
                    dim1_str = entity.get("entity_type_dim1", "")
                    dim1_labels = [k.strip() for k in dim1_str.replace('，', ',').split(',') if k.strip()]
                    if not dim1_labels:
                        dim1_labels = ["UNKNOWN"]

                    # 解析entity_type_dim2（支持中英文逗号）
                    dim2_str = entity.get("entity_type_dim2", "")
                    dim2_labels = [k.strip() for k in dim2_str.replace('，', ',').split(',') if k.strip()]
                    if not dim2_labels:
                        dim2_labels = ["UNKNOWN"]

                    # 合并所有标签
                    type_labels = dim1_labels + dim2_labels

                    # 创建节点（使用所有标签）
                    node = Node(
                               "Entity",  # 通用标签，用于向量索引
                               *type_labels,  # 解包所有标签
                               **node_props)
                    tx.create(node)
                    stats["entities_upserted"] += 1

                except Exception as e:
                    stats["entities_failed"] += 1
                    stats["errors"].append(f"实体 {entity.get('entity_id', 'unknown')} 插入失败: {str(e)}")
                    logger.error(f"插入实体失败: {e}")

            # 提交当前批次
            graph.commit(tx)
            logger.info(f"批次 {batch_idx + 1}/{len(entity_batches)} 完成: {len(entity_batch)} 个实体, 累计: {stats['entities_upserted']}")

        except Exception as e:
            # 回滚失败批次
            graph.rollback(tx)
            stats["entities_failed"] += len(entity_batch)
            stats["errors"].append(f"实体批次 {batch_idx + 1} 插入失败: {str(e)}")
            logger.error(f"实体批次 {batch_idx + 1} 插入失败: {e}")

    logger.info(f"实体插入完成: 成功 {stats['entities_upserted']}, 失败 {stats['entities_failed']}")

    # 调试：记录前10个实体ID用于对比
    logger.info("前10个实体ID示例:")
    for i, entity in enumerate(merged_entities[:10]):
        entity_id = entity.get("entity_id") or entity.get("id") or entity.get("entity_name", "")
        logger.info(f"  {i+1}. {entity_id}")

    # ==================== 预构建节点映射 ====================
    logger.info("预构建节点映射...")
    entity_id_to_node = {}
    for entity in merged_entities:
        entity_id = entity.get("entity_id") or entity.get("id") or entity.get("entity_name", "")
        if not entity_id:
            continue
        # 查找节点
        node = graph.nodes.match("Entity", entity_id=entity_id).first()
        if node:
            entity_id_to_node[entity_id] = node

    logger.info(f"节点映射完成: {len(entity_id_to_node)} 个节点")

    # ==================== 插入关系数据 ====================
    logger.info("开始插入关系数据...")

    BATCH_SIZE = 1000
    current_batch = []
    self_loops_removed = 0

    for relation in merged_relations:
        try:
            src_id = relation.get("src_id")
            tgt_id = relation.get("tgt_id")

            if not src_id or not tgt_id:
                stats["relations_failed"] += 1
                continue

            # 过滤自环关系
            if src_id == tgt_id:
                self_loops_removed += 1
                stats["relations_self_loops_removed"] += 1
                logger.debug(f"跳过自环关系: {src_id} -> {tgt_id}")
                continue

            # 从预构建映射中获取节点
            src_node = entity_id_to_node.get(src_id)
            tgt_node = entity_id_to_node.get(tgt_id)

            if not src_node or not tgt_node:
                stats["relations_failed"] += 1
                continue

            # 清洗keywords
            keywords_str = relation.get("keywords", "")
            original_desc = relation.get("description", "")
            keyword_labels, cleaned_desc = clean_relation_keywords(keywords_str, original_desc)
            rel_type = keyword_labels[0] if keyword_labels else "RELATED_TO"

            # 创建关系
            rel = Relationship(src_node, rel_type, tgt_node,
                             weight=float(relation.get("weight", 1.0)),
                             description=cleaned_desc,
                             keywords=",".join(keyword_labels),
                             keyword_labels=keyword_labels,
                             source_id=relation.get("source_chunk_id", ""),
                             file_path=relation.get("file_path", ""),
                             created_at=relation.get("created_at", int(time.time())))

            current_batch.append(rel)

            # 达到批次大小时提交
            if len(current_batch) >= BATCH_SIZE:
                tx = graph.begin()
                try:
                    for rel in current_batch:
                        tx.create(rel)
                    graph.commit(tx)
                    stats["relations_upserted"] += len(current_batch)
                except Exception as e:
                    graph.rollback(tx)
                    stats["relations_failed"] += len(current_batch)
                    logger.error(f"关系批次插入失败: {e}")
                finally:
                    current_batch = []

        except Exception as e:
            stats["relations_failed"] += 1
            logger.error(f"关系插入失败: {e}")

    # 处理剩余关系
    if current_batch:
        tx = graph.begin()
        try:
            for rel in current_batch:
                tx.create(rel)
            graph.commit(tx)
            stats["relations_upserted"] += len(current_batch)
        except Exception as e:
            graph.rollback(tx)
            stats["relations_failed"] += len(current_batch)
            logger.error(f"最终关系批次插入失败: {e}")

    logger.info(f"已创建 {stats['relations_upserted']} 个关系到Neo4j，已过滤 {self_loops_removed} 个自环关系")

    # ==================== 打印统计信息 ====================
    logger.info("="*60)
    logger.info("Neo4j存储统计:")
    logger.info(f"  实体清空: {stats['entities_cleared']}")
    logger.info(f"  关系清空: {stats['relations_cleared']}")
    logger.info(f"  实体插入成功: {stats['entities_upserted']}")
    logger.info(f"  实体插入失败: {stats['entities_failed']}")
    logger.info(f"  关系插入成功: {stats['relations_upserted']}")
    logger.info(f"  关系插入失败: {stats['relations_failed']}")
    logger.info(f"  自环关系过滤: {stats['relations_self_loops_removed']}")
    if stats["errors"]:
        logger.warning(f"  错误数量: {len(stats['errors'])}")
        for error in stats["errors"][:5]:  # 只显示前5个错误
            logger.warning(f"    - {error}")
    logger.info("="*60)

    logger.info("=== save_to_neo4j 执行完成 ===")
    return stats


def collect_merged_entities_from_results(chunk_results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从分块处理结果中收集所有合并后的实体
    
    Args:
        chunk_results: 实体分块处理结果
        
    Returns:
        List[Dict[str, Any]]: 合并后的实体列表
    """
    merged_entities = []
    
    for chunk_name, result in chunk_results.items():
        if result.get("status") != "processed":
            continue
        
        entities = result.get("merged_entities", [])
        merged_entities.extend(entities)
        
    logger.info(f"从 {len(chunk_results)} 个分块中收集了 {len(merged_entities)} 个合并实体")
    return merged_entities


def collect_merged_relations_from_results(relation_chunk_results: Dict[Tuple[str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    从关系分块处理结果中收集所有合并后的关系
    
    Args:
        relation_chunk_results: 关系分块处理结果
        
    Returns:
        List[Dict[str, Any]]: 合并后的关系列表
    """
    merged_relations = []
    
    for chunk_key, result in relation_chunk_results.items():
        if result.get("status") != "processed":
            continue
        
        relation = result.get("result")
        if relation:
            merged_relations.append(relation)
    
    logger.info(f"从 {len(relation_chunk_results)} 个分块中收集了 {len(merged_relations)} 个合并关系")
    return merged_relations


async def test_save_to_neo4j():
    """
    测试函数：从merged_data目录读取实体和关系，直接存储到Neo4j

    用途：
    1. 测试save_to_neo4j函数性能
    2. 快速将已处理的数据导入Neo4j（无需重新运行实体合并）

    使用方式：
    python entity_merge.py --test
    """
    import argparse

    # 设置命令行参数
    parser = argparse.ArgumentParser(description="测试Neo4j存储功能")
    parser.add_argument("--test", action="store_true", help="运行测试模式")
    parser.add_argument("--data-dir", default="./merged_data", help="数据目录路径")
    parser.add_argument("--clear", default="true", help="是否清空现有数据 (true/false)")
    args = parser.parse_args()

    if not args.test:
        print("使用 --test 参数运行测试模式")
        print("示例：python entity_merge.py --test --data-dir ./merged_data --clear true")
        return

    logger.info("=== 测试模式：从文件读取数据并存储到Neo4j ===")

    entities_file = os.path.join(args.data_dir, "processed_entities.json")
    relations_file = os.path.join(args.data_dir, "processed_relations.json")

    # 检查文件是否存在
    if not os.path.exists(entities_file):
        logger.error(f"实体文件不存在: {entities_file}")
        logger.info("请先运行完整的实体合并流程生成数据")
        return

    if not os.path.exists(relations_file):
        logger.warning(f"关系文件不存在: {relations_file}")
        logger.info("将只导入实体数据")

    # 读取数据
    logger.info(f"从 {args.data_dir} 读取数据...")
    with open(entities_file, 'r', encoding='utf-8') as f:
        merged_entities = json.load(f)

    merged_relations = []
    if os.path.exists(relations_file):
        with open(relations_file, 'r', encoding='utf-8') as f:
            merged_relations = json.load(f)

    # 解码 base64 embedding（与 load_existing_merged_data 保持一致）
    logger.info("解码实体嵌入向量...")
    for entity in merged_entities:
        if "embedding" in entity and isinstance(entity["embedding"], str):
            try:
                entity["embedding"] = embed_base64_to_list(entity["embedding"])
            except Exception as e:
                logger.warning(f"实体 {entity.get('entity_id', 'unknown')} 嵌入向量解码失败: {e}")
                entity["embedding"] = None

    logger.info(f"读取完成: {len(merged_entities)} 个实体, {len(merged_relations)} 个关系")

    # 调用save_to_neo4j
    clear_existing = args.clear.lower() == "true"

    logger.info(f"开始存储到Neo4j，清空模式: {clear_existing}")
    neo4j_stats = await save_to_neo4j(
        merged_entities=merged_entities,
        merged_relations=merged_relations,
        clear_existing=clear_existing
    )

    # 打印结果
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print(f"数据源: {args.data_dir}")
    print(f"Neo4j实体: {neo4j_stats['entities_upserted']}")
    print(f"Neo4j关系: {neo4j_stats['relations_upserted']}")
    print(f"失败: {neo4j_stats['entities_failed'] + neo4j_stats['relations_failed']}")
    print("="*60)


if __name__ == "__main__":
    # 导入LLM使用跟踪器
    from llm_tracker import init_tracker, cleanup

    # 初始化tracker（根据环境变量 ENABLE_LLM_TRACKING 控制）
    init_tracker()

    try:
        # 检查是否运行测试模式
        import sys
        if "--save" in sys.argv:
            asyncio.run(test_save_to_neo4j())
        else:
            # 运行数据接管
            asyncio.run(main())
    finally:
        # 程序结束时自动打印报告并卸载tracker
        cleanup()

