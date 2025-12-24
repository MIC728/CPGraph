"""
CPGraph 多线程异步文件提取模块

主要特性:
- 多线程并行处理 (ThreadPoolExecutor)
- 线程内异步并发 (asyncio)
- LLM实体关系提取
- JSON格式输出
- 进度监控和错误处理
"""

import os
import time
import json
import asyncio
import logging
import random
import string
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Callable

# LightRAG 相关导入
from lightrag.operate import chunking_by_token_size
from lightrag.utils import compute_mdhash_id, TiktokenTokenizer, setup_logger, get_env_value
from lightrag.prompt import PROMPTS


def get_default_tokenizer():
    """
    获取默认的 tokenizer 实例

    Returns:
        TiktokenTokenizer: 默认的 tiktoken tokenizer 实例
    """
    return TiktokenTokenizer("gpt-4o-mini")

# 配置日志 - 复用LightRAG logger
log_level = get_env_value("LOG_LEVEL", "INFO", str).upper()
# 是否启用文件日志（默认为true）
enable_file_logging = get_env_value("LOG_FILE_ENABLE", "true", str).lower() == "true"
# 日志文件路径（默认使用 LOG_DIR + cprag.log）
log_file_path = None
if enable_file_logging:
    log_dir = get_env_value("LOG_DIR", ".", str)
    log_file_path = os.path.abspath(os.path.join(log_dir, "cprag.log"))

setup_logger(
    logger_name="CPGraph",
    level=log_level,
    add_filter=False,
    log_file_path=log_file_path,
    enable_file_logging=enable_file_logging
)
logger = logging.getLogger("CPGraph")


@dataclass
class ExtractionConfig:
    """提取配置"""
    # 并发控制
    thread_count: int = 16  # 默认CPU核心数
    max_concurrent_per_thread: int = 24  # 每线程最大并发

    # 文本分块参数
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 128

    # LLM调用参数
    llm_max_retries: int = 3
    llm_timeout: int = 120  # 秒
    llm_retry_delay: float = 2.0  # 秒

    # 提取模式
    extraction_mode: str = "normal"  # "normal" 或 "problem"
    # 注意：题目提取模式(extraction_mode="problem")会自动将chunk_token_size设置为极大值(100000000)
    # 以避免分割题目和题解，保持实体的完整性

    # 输出参数
    output_dir: str = "../extracted_data"
    incremental_write: bool = False  # 是否增量写入（追加到现有文件），默认False（覆盖）

    # 性能调优
    enable_progress_logging: bool = True
    log_interval: int = 10  # 每处理N个文档打印一次日志

    def __post_init__(self):
        """后处理验证"""
        os.makedirs(self.output_dir, exist_ok=True)

        # 自动检测CPU核心数
        if self.thread_count is None:
            self.thread_count = min(32, (os.cpu_count() or 1) + 4)


class ProgressTracker:
    """进度跟踪器"""

    def __init__(self, total: int, log_interval: int = 10):
        self.total = total
        self.current = 0
        self.log_interval = log_interval
        self.start_time = time.time()

    def update(self, count: int = 1):
        """更新进度"""
        self.current += count

        if self.current % self.log_interval == 0 or self.current == self.total:
            elapsed = time.time() - self.start_time
            rate = self.current / elapsed if elapsed > 0 else 0

            logger.info(
                f"进度: {self.current}/{self.total} "
                f"({self.current/self.total*100:.1f}%) - "
                f"速度: {rate:.2f} 文档/秒"
            )


class WorkerThread:
    """工作线程：在线程内执行异步文档处理"""

    def __init__(self, thread_id: int, llm_func: Callable, tokenizer, config: ExtractionConfig):
        self.thread_id = thread_id
        self.llm_func = llm_func
        self.tokenizer = tokenizer
        self.config = config

    def process_document_group(self, doc_group: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """线程内处理文档组 (同步接口)"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._async_process_documents_in_thread(doc_group)
            )
        finally:
            loop.close()

    async def _async_process_documents_in_thread(self, doc_group: List[str]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """线程内异步处理文档组"""
        if not doc_group:
            return [], [], []

        semaphore = asyncio.Semaphore(self.config.max_concurrent_per_thread)

        async def process_single_doc(doc: str, doc_index: int) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            async with semaphore:
                return await self._async_process_single_doc(doc, doc_index)

        # 创建所有文档的处理任务，为每个文档分配唯一索引
        tasks = [process_single_doc(doc, i) for i, doc in enumerate(doc_group)]

        # 异步并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 聚合结果
        all_entities = []
        all_relations = []
        all_chunks = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning(f"线程{self.thread_id}文档处理失败: {result}")
                continue

            entities, relations, chunks = result
            all_entities.extend(entities)
            all_relations.extend(relations)
            all_chunks.extend(chunks)

        return all_entities, all_relations, all_chunks

    async def _async_process_single_doc(self, doc: str, doc_index: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """异步处理单个文档"""
        try:
            # 为每个文档生成基于完整内容的file_path，确保唯一性
            import hashlib
            # 基于完整文档内容生成hash，保证唯一性
            file_path_hash = hashlib.md5(doc.encode('utf-8')).hexdigest()[:16]
            file_path = f"doc-{file_path_hash}"

            # 1. 文本分块
            chunks = self._chunk_text(doc, file_path)

            # 2. LLM实体关系提取
            entities, relations = await self._extract_entities_with_llm(chunks)

            return entities, relations, chunks

        except Exception as e:
            logger.error(f"线程{self.thread_id}文档处理失败: {e}")
            return [], [], []

    def _chunk_text(self, text: str, file_path: str = "extracted_document") -> List[Dict[str, Any]]:
        """文本分块 (复用 LightRAG 逻辑)"""
        # 检测并处理JSON格式输入（题目等数据）
        if text.strip().startswith(('{', '[')):
            try:
                data = json.loads(text)
                # 从JSON中提取所有文本内容
                if isinstance(data, dict):
                    text = ' '.join(str(v) for v in data.values() if isinstance(v, (str, int, float)))
                elif isinstance(data, list):
                    text = ' '.join(str(v) for v in data if isinstance(v, (str, int, float)))
            except:
                pass  # 如果解析失败，继续使用原始文本

        chunk_dicts = chunking_by_token_size(
            tokenizer=self.tokenizer,
            content=text,
            split_by_character=None,
            split_by_character_only=False,
            overlap_token_size=self.config.chunk_overlap_token_size,
            max_token_size=self.config.chunk_token_size
        )

        # 转换为标准格式
        chunks = []
        for i, chunk_data in enumerate(chunk_dicts):
            chunk_id = compute_mdhash_id(chunk_data["content"], prefix="chunk-")
            chunks.append({
                "chunk_id": chunk_id,
                "content": chunk_data["content"],
                "tokens": chunk_data["tokens"],
                "chunk_order_index": i,
                "full_doc_id": f"doc-{self.thread_id}",  # 临时ID
                "file_path": file_path,  # 使用传入的文档路径
            })

        return chunks

    async def _extract_entities_with_llm(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
        """LLM实体关系提取 - chunk间异步并发"""
        if not chunks:
            return [], []

        async def extract_single_chunk(chunk: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
            try:
                entities, relations = await self._call_llm_with_retry(chunk)
                logger.info(f"[线程{self.thread_id}] Chunk {chunk['chunk_id'][:20]}... 完成: {len(entities)}实体, {len(relations)}关系")
                return entities, relations
            except Exception as e:
                logger.warning(f"[线程{self.thread_id}] Chunk {chunk['chunk_id'][:20]}... 失败: {e}")
                return [], []

        # 并发处理所有 chunks
        tasks = [extract_single_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks)

        # 聚合结果
        all_entities = []
        all_relations = []
        for entities, relations in results:
            all_entities.extend(entities)
            all_relations.extend(relations)

        return all_entities, all_relations

    async def _call_llm_with_retry(self, chunk: Dict[str, Any], max_retries: int = None) -> Tuple[List[Dict], List[Dict]]:
        """带重试机制的LLM调用"""
        max_retries = max_retries or self.config.llm_max_retries
        retry_delay = self.config.llm_retry_delay

        last_error = None

        for attempt in range(max_retries):
            try:
                # 构建prompt
                prompt = self._build_entity_extraction_prompt(chunk)

                # 调用LLM (带超时)
                response = await asyncio.wait_for(
                    self.llm_func(prompt),
                    timeout=self.config.llm_timeout
                )

                # 解析结果
                entities, relations = self._parse_llm_response(response, chunk)
                return entities, relations

            except asyncio.TimeoutError:
                last_error = f"LLM调用超时 (尝试 {attempt + 1}/{max_retries})"
                logger.warning(last_error)

            except Exception as e:
                last_error = f"LLM调用失败: {str(e)}"
                logger.warning(last_error)

                # 判断是否可重试
                if not self._is_retryable_error(e):
                    break

            # 指数退避重试
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                await asyncio.sleep(wait_time)

        # 所有重试都失败，返回空结果
        logger.error(f"LLM调用最终失败: {last_error}")
        return [], []

    def _build_entity_extraction_prompt(self, chunk: Dict[str, Any]) -> str:
        """构建实体提取prompt"""
        # 定义实体类型
        entity_types_dim1 = "数据结构, 算法思想, 动态规划, 图论, 搜索, 字符串, 数学, 计算几何, 其他"
        entity_types_dim2 = "概念, 技巧, 实现, 模型, 算法, 原理, 题目, 题解, 其他"

        # 获取分隔符
        tuple_delimiter = PROMPTS.get("DEFAULT_TUPLE_DELIMITER", "<|#|>")
        completion_delimiter = PROMPTS.get("DEFAULT_COMPLETION_DELIMITER", "<|COMPLETE|>")

        # 根据提取模式选择prompt模板
        if self.config.extraction_mode == "problem":
            # 题目专用提取模式（使用极大chunk_size避免分割题目和题解）
            if hasattr(self.config, '_problem_mode_initialized'):
                # 已经初始化过，直接使用
                pass
            else:
                # 第一次检测到题目模式，设置为极大chunk_size
                self.config.chunk_token_size = 100000000  # 1亿token，几乎不会分割
                self.config.chunk_overlap_token_size = 0
                self.config._problem_mode_initialized = True
            system_prompt = PROMPTS["problem_entity_extraction_system_prompt"].format(
                input_text=chunk["content"],
                entity_types_dim1=entity_types_dim1,
                entity_types_dim2=entity_types_dim2,
                tuple_delimiter=tuple_delimiter,
                completion_delimiter=completion_delimiter,
                language="zh-CN",
                examples=PROMPTS.get("problem_entity_extraction_examples", "")
            )

            user_prompt = PROMPTS["problem_entity_extraction_user_prompt"].format(
                tuple_delimiter=tuple_delimiter,
                completion_delimiter=completion_delimiter,
                language="zh-CN"
            )
        else:
            # 常规提取模式（复用原有逻辑）
            examples = PROMPTS.get("entity_extraction_examples", "")

            system_prompt = PROMPTS["entity_extraction_system_prompt"].format(
                input_text=chunk["content"],
                entity_types_dim1=entity_types_dim1,
                entity_types_dim2=entity_types_dim2,
                tuple_delimiter=tuple_delimiter,
                completion_delimiter=completion_delimiter,
                language="zh-CN",
                examples=examples
            )

            user_prompt = PROMPTS["entity_extraction_user_prompt"].format(
                input_text=chunk["content"],
                tuple_delimiter=tuple_delimiter,
                completion_delimiter=completion_delimiter,
                language="zh-CN",
                entity_types_dim1=entity_types_dim1,
                entity_types_dim2=entity_types_dim2,
                examples=examples
            )

        # 组合prompt
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        return full_prompt

    def _parse_llm_response(self, response: str, chunk: Dict[str, Any]) -> Tuple[List[Dict], List[Dict]]:
        """解析LLM响应"""
        entities = []
        relations = []

        try:
            # 解析实体 - 每个实体以"entity<|#|>"开头，以换行符或字符串结尾结束
            entity_pattern = r"entity<\|#\|>(.*?)(?=\n(?:entity|relation)<\|#\|>|$)"
            entity_matches = re.findall(entity_pattern, response, re.DOTALL)

            for match in entity_matches:
                parts = [p.strip() for p in match.split("<|#|>")]
                if len(parts) >= 4:
                    entity = {
                        "entity_name": parts[0],
                        "entity_type_dim1": parts[1] if len(parts) > 1 else "",
                        "entity_type_dim2": parts[2] if len(parts) > 2 else "",
                        "description": parts[3] if len(parts) > 3 else "",
                        "source_id": chunk["chunk_id"],
                        "file_path": chunk.get("file_path", "unknown"),
                        "data_source": "extracted",
                        "is_problem_extracted": self.config.extraction_mode == "problem"  # 添加题目提取标记
                    }
                    entities.append(entity)

            # 解析关系 - 每个关系以"relation<|#|>"开头，以换行符或字符串结尾结束
            relation_pattern = r"relation<\|#\|>(.*?)(?=\n(?:entity|relation)<\|#\|>|$)"
            relation_matches = re.findall(relation_pattern, response, re.DOTALL)

            for match in relation_matches:
                parts = [p.strip() for p in match.split("<|#|>")]
                if len(parts) >= 3:
                    # 清理 description 中可能残留的 <|COMPLETE|> 标记
                    description = parts[3] if len(parts) > 3 else ""
                    description = description.replace("<|COMPLETE|>", "").strip()

                    relation = {
                        "src_name": parts[0],  # 使用名称而不是ID
                        "tgt_name": parts[1],  # 使用名称而不是ID
                        "keywords": parts[2] if len(parts) > 2 else "",
                        "description": description,
                        "weight": 1.0,
                        "source_chunk_id": chunk["chunk_id"],
                        "file_path": chunk.get("file_path", "unknown"),
                        "data_source": "extracted"
                    }
                    relations.append(relation)

        except Exception as e:
            logger.warning(f"解析LLM响应失败: {e}")

        # 在当前chunk内合并同名实体
        entities = self._merge_chunk_entities(entities)
        # 注意：关系解析和ID分配移到文档级别进行

        return entities, relations

    def _merge_chunk_entities(self, entities: List[Dict]) -> List[Dict]:
        """在chunk内合并同名实体"""
        if not entities:
            return []

        entity_map = {}
        for entity in entities:
            name = entity["entity_name"]
            if name not in entity_map:
                # 新实体
                entity_map[name] = entity.copy()
            else:
                # 同名实体：合并类型和描述
                existing = entity_map[name]
                # 合并类型（去重）
                for dim in ["entity_type_dim1", "entity_type_dim2"]:
                    existing_types = set(existing.get(dim, "").split(",") if existing.get(dim) else [])
                    new_types = set(entity.get(dim, "").split(",") if entity.get(dim) else [])
                    merged = list(existing_types | new_types)
                    if merged:
                        existing[dim] = ",".join(merged)
                    else:
                        existing[dim] = ""
                # 合并描述（去重）
                existing_desc = existing.get("description", "")
                new_desc = entity.get("description", "")
                if new_desc and new_desc not in existing_desc:
                    existing["description"] = f"{existing_desc}<SEP>{new_desc}".strip()

        return list(entity_map.values())

    def _resolve_relations_by_doc(self, relations: List[Dict], entities: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        按文档分组解析关系：确保每个文档内的关系只引用该文档内的实体
        如果实体不存在，创建 UNKNOWN 类型的新实体

        Returns:
            Tuple[List[Dict], List[Dict]]: (resolved_relations, new_entities)
        """
        if not relations:
            return [], []

        # 按file_path分组处理，确保文档级实体管理
        doc_groups = {}
        for entity in entities:
            file_path = entity.get("file_path", "unknown")
            if file_path not in doc_groups:
                doc_groups[file_path] = []
            doc_groups[file_path].append(entity)

        all_resolved_relations = []
        all_new_entities = []

        # 为每个文档组单独解析关系
        for file_path, doc_entities in doc_groups.items():
            # 构建当前文档的实体名称映射
            name_to_entity = {e["entity_name"]: e for e in doc_entities}
            doc_new_entities = []

            def get_or_create_entity(name: str, relation: Dict) -> Dict:
                """获取或创建实体（仅在当前文档内）"""
                if name in name_to_entity:
                    return name_to_entity[name]

                # 创建 UNKNOWN 类型的新实体
                new_entity = {
                    "entity_name": name,
                    "entity_type_dim1": "UNKNOWN",
                    "entity_type_dim2": "UNKNOWN",
                    "description": relation.get("description", ""),
                    "source_id": relation.get("source_chunk_id", ""),
                    "file_path": file_path,  # 使用当前文档的file_path
                    "data_source": "inferred"  # 标记为推断生成
                }
                name_to_entity[name] = new_entity
                doc_new_entities.append(new_entity)
                logger.info(f"文档 '{file_path}' 创建推断实体: {name}")
                return new_entity

            # 解析当前文档的所有关系
            for relation in relations:
                # 只处理属于当前文档的关系
                if relation.get("file_path") != file_path:
                    continue

                src_name = relation.get("src_name")
                tgt_name = relation.get("tgt_name")

                if not src_name or not tgt_name:
                    continue

                # 获取或创建实体（仅在当前文档内）
                src_entity = get_or_create_entity(src_name, relation)
                tgt_entity = get_or_create_entity(tgt_name, relation)

                resolved_relation = relation.copy()
                resolved_relation["src_entity"] = src_entity
                resolved_relation["tgt_entity"] = tgt_entity
                # 删除临时字段
                resolved_relation.pop("src_name", None)
                resolved_relation.pop("tgt_name", None)
                all_resolved_relations.append(resolved_relation)

            all_new_entities.extend(doc_new_entities)

        return all_resolved_relations, all_new_entities

    def _assign_random_ids(self, entities: List[Dict], relations: List[Dict]):
        """一次性为所有实体和关系分配随机ID"""
        # 为每个实体分配唯一ID
        for entity in entities:
            entity["entity_id"] = self._generate_entity_id(entity["entity_name"])

        # 为每个关系分配ID
        for relation in relations:
            src_entity = relation.get("src_entity")
            tgt_entity = relation.get("tgt_entity")
            if src_entity and tgt_entity:
                relation["src_id"] = src_entity["entity_id"]
                relation["tgt_id"] = tgt_entity["entity_id"]
                # 删除临时字段
                relation.pop("src_entity", None)
                relation.pop("tgt_entity", None)

    def _generate_entity_id(self, entity_name: str) -> str:
        """为实体名称生成带随机ID的实体ID"""
        random_suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        return f"{entity_name}<{random_suffix}>"

    def _is_retryable_error(self, error: Exception) -> bool:
        """判断错误是否可重试"""
        error_str = str(error).lower()

        # 不可重试的错误
        non_retryable_keywords = ['503', 'unauthorized', 'forbidden', 'quota']

        if any(keyword in error_str for keyword in non_retryable_keywords):
            return False

        # 可重试的错误
        retryable_keywords = ['timeout', 'connection', 'network', '502', '504']
        return any(keyword in error_str for keyword in retryable_keywords)


class FileExtractionController:
    """主控制器：管理多线程异步文档提取"""

    def __init__(self, llm_func: Callable, tokenizer, config: ExtractionConfig = None):
        """
        初始化控制器

        Args:
            llm_func: LLM调用函数
            tokenizer: LightRAG tokenizer
            config: 提取配置
        """
        self.llm_func = llm_func
        self.tokenizer = tokenizer
        self.config = config or ExtractionConfig()

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)

        logger.info(f"初始化文件提取控制器:")
        logger.info(f"  线程数: {self.config.thread_count}")
        logger.info(f"  每线程并发数: {self.config.max_concurrent_per_thread}")
        logger.info(f"  输出目录: {self.config.output_dir}")

    async def extract_documents(self, documents: List[str]) -> Tuple[List[Dict], List[Dict]]:
        """
        主入口：提取文档中的实体和关系

        Args:
            documents: 文档列表

        Returns:
            Tuple[List[Dict], List[Dict]]: (entities, relations)
        """
        if not documents:
            logger.warning("没有提供文档，返回空结果")
            return [], []

        start_time = time.time()
        logger.info(f"开始提取 {len(documents)} 个文档的实体和关系...")

        # 1. 分组文档
        doc_groups = self._split_documents(documents, self.config.thread_count)
        logger.info(f"文档分组: {len(doc_groups)} 组")

        # 2. 创建线程池并提交任务
        results = []
        progress = ProgressTracker(len(documents), self.config.log_interval)

        with ThreadPoolExecutor(max_workers=self.config.thread_count) as executor:
            futures = []

            for i, doc_group in enumerate(doc_groups):
                if not doc_group:  # 跳过空组
                    continue

                worker = WorkerThread(i, self.llm_func, self.tokenizer, self.config)
                future = executor.submit(worker.process_document_group, doc_group)
                futures.append(future)

            # 3. 等待所有任务完成
            for future in as_completed(futures):
                try:
                    entities, relations, chunks = future.result()
                    results.append((entities, relations, chunks))
                    progress.update(len(doc_group))

                except Exception as e:
                    logger.error(f"线程任务失败: {e}")
                    results.append(([], [], []))

        # 4. 聚合结果
        all_entities = []
        all_relations = []
        all_chunks = []

        for entities, relations, chunks in results:
            all_entities.extend(entities)
            all_relations.extend(relations)
            all_chunks.extend(chunks)

        # 5. 文档级别：合并同名实体
        all_entities = self._merge_entities_by_doc(all_entities)

        # 6. 文档级别：解析关系（确保每个文档内的关系只引用该文档内的实体）
        resolver = WorkerThread(0, self.llm_func, self.tokenizer, self.config)
        all_relations, new_entities = resolver._resolve_relations_by_doc(all_relations, all_entities)
        all_entities.extend(new_entities)

        # 7. 文档级别：分配随机ID
        resolver._assign_random_ids(all_entities, all_relations)

        # 8. 移除 entity_name（已有 entity_id）
        for entity in all_entities:
            entity.pop("entity_name", None)

        # 9. 去重 (基于entity_id和chunk_id)
        all_entities = self._deduplicate_entities(all_entities)
        all_relations = self._deduplicate_relations(all_relations)
        all_chunks = self._deduplicate_chunks(all_chunks)

        # 10. 保存到JSON（包括chunks）
        await self._save_to_json(all_entities, all_relations, all_chunks)

        # 11. 输出统计信息
        elapsed_time = time.time() - start_time
        speed = len(documents) / elapsed_time if elapsed_time > 0 else 0

        inferred_count = sum(1 for e in all_entities if e.get("data_source") == "inferred")
        logger.info(f"提取完成:")
        logger.info(f"  处理文档数: {len(documents)}")
        logger.info(f"  提取实体数: {len(all_entities)} (其中推断实体: {inferred_count})")
        logger.info(f"  提取关系数: {len(all_relations)}")
        logger.info(f"  提取分块数: {len(all_chunks)}")
        logger.info(f"  总耗时: {elapsed_time:.2f}秒")
        logger.info(f"  处理速度: {speed:.2f} 文档/秒")

        return all_entities, all_relations

    def _split_documents(self, documents: List[str], group_count: int) -> List[List[str]]:
        """将文档列表分成指定数量的组，尽量均匀分配"""
        if group_count >= len(documents):
            # 如果线程数大于文档数，每个文档一组
            return [[doc] for doc in documents]

        # 计算每组的基本大小
        base_group_size = len(documents) // group_count
        extra_docs = len(documents) % group_count

        groups = []
        start_idx = 0

        for i in range(group_count):
            # 前 extra_docs 组多分配一个文档
            group_size = base_group_size + (1 if i < extra_docs else 0)

            if group_size > 0:
                end_idx = start_idx + group_size
                groups.append(documents[start_idx:end_idx])
                start_idx = end_idx
            else:
                groups.append([])

        return groups

    def _merge_entities_by_doc(self, entities: List[Dict]) -> List[Dict]:
        """按文档合并同名实体（同一文档内的不同chunk）"""
        if not entities:
            return []

        # 按file_path分组
        doc_groups = {}
        for entity in entities:
            file_path = entity.get("file_path", "unknown")
            if file_path not in doc_groups:
                doc_groups[file_path] = []
            doc_groups[file_path].append(entity)

        merged_entities = []
        total_merged = 0

        for file_path, doc_entities in doc_groups.items():
            # 在同一文档内合并同名实体
            entity_map = {}
            for entity in doc_entities:
                name = entity.get("entity_name")
                if not name:
                    continue

                if name not in entity_map:
                    entity_map[name] = entity.copy()
                else:
                    # 同文档内合并：合法
                    existing = entity_map[name]
                    # 合并类型（去重）
                    for dim in ["entity_type_dim1", "entity_type_dim2"]:
                        existing_types = set(existing.get(dim, "").split(",") if existing.get(dim) else [])
                        new_types = set(entity.get(dim, "").split(",") if entity.get(dim) else [])
                        merged = [t for t in (existing_types | new_types) if t]
                        existing[dim] = ",".join(merged) if merged else ""
                    # 合并描述
                    existing_desc = existing.get("description", "")
                    new_desc = entity.get("description", "")
                    if new_desc and new_desc not in existing_desc:
                        existing["description"] = f"{existing_desc}<SEP>{new_desc}".strip("<SEP>")
                    # 合并source_id
                    existing_source = existing.get("source_id", "")
                    new_source = entity.get("source_id", "")
                    if new_source and new_source not in existing_source:
                        existing["source_id"] = f"{existing_source}<SEP>{new_source}".strip("<SEP>")

            doc_merged_count = len(doc_entities) - len(entity_map)
            total_merged += doc_merged_count
            merged_entities.extend(entity_map.values())

        if total_merged > 0:
            logger.info(f"实体合并: {len(entities)} -> {len(merged_entities)} (在文档内合并了 {total_merged} 个重复实体)")

        return merged_entities

    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """基于entity_id去重实体"""
        seen = set()
        deduplicated = []

        for entity in entities:
            entity_id = entity.get("entity_id")
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                deduplicated.append(entity)

        if len(entities) != len(deduplicated):
            logger.info(f"实体去重: {len(entities)} -> {len(deduplicated)}")

        return deduplicated

    def _deduplicate_relations(self, relations: List[Dict]) -> List[Dict]:
        """基于(src_id, tgt_id)去重关系"""
        seen = set()
        deduplicated = []

        for relation in relations:
            key = (relation.get("src_id"), relation.get("tgt_id"))
            if key not in seen:
                seen.add(key)
                deduplicated.append(relation)

        if len(relations) != len(deduplicated):
            logger.info(f"关系去重: {len(relations)} -> {len(deduplicated)}")

        return deduplicated

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """基于chunk_id去重分块"""
        seen = set()
        deduplicated = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            if chunk_id and chunk_id not in seen:
                seen.add(chunk_id)
                deduplicated.append(chunk)

        if len(chunks) != len(deduplicated):
            logger.info(f"分块去重: {len(chunks)} -> {len(deduplicated)}")

        return deduplicated

    async def _save_to_json(self, entities: List[Dict], relations: List[Dict], chunks: List[Dict] = None):
        """保存到JSON文件"""
        try:
            # 统一使用相同的文件名
            entities_file = os.path.join(self.config.output_dir, "entities.json")
            relations_file = os.path.join(self.config.output_dir, "relations.json")
            chunks_file = os.path.join(self.config.output_dir, "chunks.json")

            # 根据配置决定写入模式
            if self.config.incremental_write:
                # 增量写入：读取现有文件，合并数据
                existing_entities = []
                existing_relations = []
                existing_chunks = {}

                # 读取现有实体
                if os.path.exists(entities_file):
                    with open(entities_file, 'r', encoding='utf-8') as f:
                        existing_entities = json.load(f)
                    logger.info(f"读取现有实体: {len(existing_entities)} 个")

                # 读取现有关系
                if os.path.exists(relations_file):
                    with open(relations_file, 'r', encoding='utf-8') as f:
                        existing_relations = json.load(f)
                    logger.info(f"读取现有关系: {len(existing_relations)} 个")

                # 读取现有分块
                if chunks and os.path.exists(chunks_file):
                    with open(chunks_file, 'r', encoding='utf-8') as f:
                        existing_chunks = json.load(f)
                    logger.info(f"读取现有分块: {len(existing_chunks)} 个")

                # 合并数据
                # 合并实体（基于entity_id去重）
                existing_entity_ids = {e.get("entity_id") for e in existing_entities}
                new_entities = [e for e in entities if e.get("entity_id") not in existing_entity_ids]
                merged_entities = existing_entities + new_entities

                # 合并关系（基于(src_id, tgt_id)去重）
                existing_relation_keys = {(r.get("src_id"), r.get("tgt_id")) for r in existing_relations}
                new_relations = [r for r in relations if (r.get("src_id"), r.get("tgt_id")) not in existing_relation_keys]
                merged_relations = existing_relations + new_relations

                # 合并分块
                if chunks:
                    new_chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
                    merged_chunks = {**existing_chunks, **new_chunks}
                else:
                    merged_chunks = existing_chunks

                logger.info(f"增量写入统计:")
                logger.info(f"  新增实体: {len(new_entities)}")
                logger.info(f"  新增关系: {len(new_relations)}")
                logger.info(f"  新增分块: {len(merged_chunks) - len(existing_chunks)}")

                # 保存合并后的数据
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_entities, f, ensure_ascii=False, indent=2)

                with open(relations_file, 'w', encoding='utf-8') as f:
                    json.dump(merged_relations, f, ensure_ascii=False, indent=2)

                if chunks:
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(merged_chunks, f, ensure_ascii=False, indent=2)

                logger.info(f"增量数据已保存到:")
                logger.info(f"  实体: {entities_file} (总计: {len(merged_entities)})")
                logger.info(f"  关系: {relations_file} (总计: {len(merged_relations)})")
                if chunks:
                    logger.info(f"  分块: {chunks_file} (总计: {len(merged_chunks)})")

            else:
                # 覆盖写入：直接保存新数据
                with open(entities_file, 'w', encoding='utf-8') as f:
                    json.dump(entities, f, ensure_ascii=False, indent=2)

                with open(relations_file, 'w', encoding='utf-8') as f:
                    json.dump(relations, f, ensure_ascii=False, indent=2)

                if chunks:
                    chunks_map = {chunk["chunk_id"]: chunk for chunk in chunks}
                    with open(chunks_file, 'w', encoding='utf-8') as f:
                        json.dump(chunks_map, f, ensure_ascii=False, indent=2)

                logger.info(f"数据已保存到:")
                logger.info(f"  实体: {entities_file}")
                logger.info(f"  关系: {relations_file}")
                if chunks:
                    logger.info(f"  分块: {chunks_file}")

        except Exception as e:
            logger.error(f"保存JSON文件失败: {e}")
            raise


# 便捷函数
async def extract_documents_async(
    documents: List[str],
    llm_func: Callable,
    tokenizer=None,
    config: ExtractionConfig = None
) -> Tuple[List[Dict], List[Dict]]:
    """
    异步提取文档的便捷函数

    Args:
        documents: 文档列表
        llm_func: LLM调用函数
        tokenizer: LightRAG tokenizer (可选，将使用默认)
        config: 提取配置

    Returns:
        Tuple[List[Dict], List[Dict]]: (entities, relations)
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()
    controller = FileExtractionController(llm_func, tokenizer, config)
    return await controller.extract_documents(documents)


async def extract_problems_async(
    documents: List[str],
    llm_func: Callable,
    tokenizer=None,
    config: ExtractionConfig = None,
    **kwargs
) -> Tuple[List[Dict], List[Dict]]:
    """
    异步提取题目相关实体的便捷函数

    Args:
        documents: 题解文档列表
        llm_func: LLM调用函数
        tokenizer: LightRAG tokenizer (可选，将使用默认)
        config: 提取配置（可选，将自动创建题目专用配置）
        **kwargs: 额外配置参数，将覆盖默认配置

    Returns:
        Tuple[List[Dict], List[Dict]]: (entities, relations)
    """
    if tokenizer is None:
        tokenizer = get_default_tokenizer()

    # 创建题目专用配置
    if config is None:
        config = ExtractionConfig()

    # 应用题目专用模式和额外参数
    config.extraction_mode = "problem"
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    controller = FileExtractionController(llm_func, tokenizer, config)
    return await controller.extract_documents(documents)


