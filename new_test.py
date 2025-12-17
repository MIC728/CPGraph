"""
LightRAG 新架构测试文件 - 使用多线程异步提取

承接原 test.py 的功能，但使用新的多线程异步架构。
支持文档插入、计时、性能测试等功能。

新架构优势:
- 真正的多线程并行 (ThreadPoolExecutor)
- 线程内异步并发 (asyncio)
- 预期性能提升 5-10 倍
- 简洁的 JSON 输出格式
"""

import os
import asyncio
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 新架构导入
from multi_threaded_extractor import extract_documents_async, extract_problems_async, ExtractionConfig, get_default_tokenizer

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ⚠️ 重要：必须在导入openai_complete_if_cache之后立即安装Hook
# 否则已导入的函数引用不会被更新，统计功能将失效
from llm_tracker import init_tracker
init_tracker()


async def create_llm_func():
    """创建 LLM 调用函数"""
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        # ⚠️ 重要：从模块动态获取函数引用，确保Hook生效
        from lightrag.llm.openai import openai_complete_if_cache
        return await openai_complete_if_cache(
            os.getenv("LLM_MODEL", "deepseek-chat"),
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            api_key=os.getenv("LLM_BINDING_API_KEY"),
            base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
            **kwargs,
        )
    return llm_model_func


def extract_md_file(filepath: str) -> str:
    """从指定文件路径提取.md文件返回字符串内容"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"文件未找到: {filepath}")
        return ""
    except Exception as e:
        logger.error(f"读取文件失败 {filepath}: {e}")
        return ""


async def measure_insert_performance_new(
    documents: List[str],
    config: ExtractionConfig = None,
    llm_func=None
) -> Dict[str, Any]:
    """
    测量新架构的插入性能

    Args:
        documents: 文档列表
        config: 提取配置
        llm_func: LLM调用函数

    Returns:
        Dict: 性能统计结果
    """
    print(f"\n开始性能测试 (新架构)，文档数量: {len(documents)}")

    # 性能测试
    start_time = time.time()

    # 使用新架构提取
    entities, relations = await extract_documents_async(
        documents=documents,
        llm_func=llm_func,
        config=config or ExtractionConfig()
    )

    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_doc = total_time / len(documents) if documents else 0

    result = {
        "total_docs": len(documents),
        "extracted_entities": len(entities),
        "extracted_relations": len(relations),
        "total_time": total_time,
        "avg_time_per_doc": avg_time_per_doc,
        "docs_per_second": len(documents) / total_time if total_time > 0 else 0,
        "entities_per_second": len(entities) / total_time if total_time > 0 else 0,
        "extraction_speed": len(documents) / total_time if total_time > 0 else 0,
        "output_files": {
            "entities": os.path.join(config.output_dir if config else "./extracted_data", "entities.json"),
            "relations": os.path.join(config.output_dir if config else "./extracted_data", "relations.json")
        }
    }

    print(f"批量插入完成 (新架构)!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个文档: {avg_time_per_doc:.2f}秒")
    print(f"处理速度: {result['docs_per_second']:.2f} 文档/秒")
    print(f"实体提取速度: {result['entities_per_second']:.2f} 实体/秒")

    return result



async def main():
    """主函数 - 承接原 test.py 的功能"""

    print("="*80)
    print("LightRAG 新架构测试 - 多线程异步文档提取")
    print("="*80)

    # 配置计时器（如果需要）
    if os.getenv("LIGHTRAG_ENABLE_TIMING", "false").lower() == "true":
        print("[OK] 计时功能已启用")

    # 配置参数
    config = ExtractionConfig(
        thread_count=int(os.getenv("THREAD_COUNT", "16")),  # 16线程
        max_concurrent_per_thread=int(os.getenv("MAX_CONCURRENT", "8")),  # 每线程8并发
        chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP", "128")),
        llm_max_retries=int(os.getenv("LLM_RETRIES", "3")),
        llm_timeout=int(os.getenv("LLM_TIMEOUT", "120")),
        output_dir="./kg_storage",
        enable_progress_logging=True,
        log_interval=5
    )

    print(f"\n配置参数:")
    print(f"  线程数: {config.thread_count}")
    print(f"  每线程并发数: {config.max_concurrent_per_thread}")
    print(f"  分块大小: {config.chunk_token_size}")
    print(f"  输出目录: {config.output_dir}")

    rag = None
    try:
        # 1. 批量插入文本 (新架构)
        print("\n" + "="*60)
        print("1. 批量文档插入 (新架构)")
        print("="*60)

        print("加载测试文件...")
        texts = [
            # 从文件路径读取
            extract_md_file("./testfile/graph/concept.md"),
            extract_md_file("./testfile/graph/mst.md"),
            # extract_md_file("./testfile/graph/shortest-path.md"),
            # extract_md_file("./testfile/graph/matrix-tree.md"),
            # extract_md_file("./testfile/ds/treap.md"),
            # extract_md_file("./testfile/ds/splay.md"),
            # extract_md_file("./testfile/ds/lct.md"),
            # extract_md_file("./testfile/ds/top-tree.md"),
            # extract_md_file("./testfile/ds/seg.md"),
            # extract_md_file("./testfile/ds/rbtree.md"),
            # extract_md_file("./testfile/ds/heap.md"),
            # "旋转是一种在树结构中调整节点位置以维持平衡的操作。",
            # # 直接文本
            "旋转是几何学中的基本图形变换，指平面内图形绕定点按某方向转动角度的运动过程，该定点称为旋转中心，转动的角度称为旋转角。二维平面是的点旋转可以使用旋转矩阵计算",
        ]
        import glob

        def get_all_files_glob(folder_path):
            """使用glob模块获取文件路径"""
            # 递归获取所有文件
            pattern = os.path.join(folder_path, "**", "*")
            all_files = glob.glob(pattern, recursive=True)

            # 过滤掉目录，只保留文件
            all_files = [f for f in all_files if os.path.isfile(f)]

            return all_files
        # for file in get_all_files_glob(r"D:\OIdata\RAG\data_process\out_data"):
        #     texts.append(extract_md_file(file))

        with open(r"D:\OIdata\server\data.json", 'r', encoding='utf-8') as file:
            data = json.load(file)
        data = data['problems']
        problem_texts = [
            # 添加一些具体的题目示例

        ]
        map = dict()
        for problem in data:
            map[problem['problem_id']] = problem
        all_ids = []
        problem_ids = []
        for id, problem in list(map.items()):
            all_ids.append(id)
        import random
        import re
        def remove_code_blocks(text): return re.sub(r'```[\s\S]*?```', '', text)
        # problem_ids = all_ids[4000:5000]
        #problem_ids = random.sample(all_ids, 30)
        # for problem in data[:300]:
        #     # if '数学' in problem.get('tags',[]) or '组合数学' in problem.get('tags',[]):
        #     problem_texts.append(json.dumps(problem, ensure_ascii=False))
        for problem_id in problem_ids:
            problem_data = map.get(problem_id)

            # 处理solutions字段：修复IndexError并支持合并多篇solution
            solutions = problem_data.get('solutions', [])
            if not solutions:
                # 如果solutions为空，使用空字符串
                final_solution = ""
            else:
                # 获取第一篇solution并去除代码块
                first_solution = solutions[0]
                if isinstance(first_solution, dict):
                    final_solution = first_solution.get('content', '') or first_solution.get('text', '') or str(first_solution)
                else:
                    final_solution = str(first_solution)
                final_solution = remove_code_blocks(final_solution)

                # 去代码后若少于500字符且存在第二篇，则合并
                if len(final_solution) < 500 and len(solutions) > 1:
                    second_solution = solutions[1]
                    if isinstance(second_solution, dict):
                        second_content = second_solution.get('content', '') or second_solution.get('text', '') or str(second_solution)
                    else:
                        second_content = str(second_solution)
                    final_solution += "\n\n" + remove_code_blocks(second_content)

            user_data = {
                'problem_id': problem_id,
                'title': problem_data.get('title'),
                'background': problem_data.get('background'),
                'description': problem_data.get('description'),
                'input_format': problem_data.get('input_format'),
                'output_format': problem_data.get('output_format'),
                'hint': problem_data.get('hint'),
                'samples': problem_data.get('samples', []),
                'difficulty': problem_data.get('difficulty'),
                'tags': problem_data.get('tags', []),
                'solution': remove_code_blocks(final_solution),
            }
            problem_texts.append(json.dumps(user_data))
        print(f"题目文本数：{len(problem_texts)}")
        # 过滤空文本
        texts = [text for text in texts if text.strip()]
        print(f"成功加载 {len(texts)} 个有效测试文件")
        print(f"文件大小统计: {[len(text) for text in texts]}")

        # 创建 LLM 函数 (在主函数中创建一次，全局使用)
        llm_func = await create_llm_func()

        # 性能测试
        performance_result = await measure_insert_performance_new(texts, config, llm_func)

        # 1.5. 题目提取测试 (新增)
        print("\n" + "="*60)
        print("1.5. 题目专用提取测试")
        print("="*60)


        # 过滤空文本
        problem_texts = [text for text in problem_texts if text.strip()]
        print(f"成功加载 {len(problem_texts)} 个题目相关文档")

        # 使用题目专用提取模式（与常规提取使用相同输出目录和文件名）
        # 注意：题目提取时会自动将chunk_size设置为极大值，避免分割题目和题解
        problem_config = ExtractionConfig(
            thread_count=int(os.getenv("THREAD_COUNT", "16")),
            max_concurrent_per_thread=int(os.getenv("MAX_CONCURRENT", "8")),
            chunk_token_size=10000000000,
            chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP", "128")),
            extraction_mode="problem",  # 启用题目专用模式
            output_dir=config.output_dir,  # 使用相同的输出目录和文件名
            enable_progress_logging=True,
            log_interval=5
        )

        print(f"\n题目提取配置:")
        print(f"  提取模式: {problem_config.extraction_mode}")
        print(f"  线程数: {problem_config.thread_count}")

        # 执行题目提取
        print(f"\n开始题目专用提取...")
        try:
            problem_entities, problem_relations = await extract_problems_async(
                documents=problem_texts,
                llm_func=llm_func,
                config=problem_config
            )

            print(f"\n题目提取完成!")
            print(f"  提取实体数: {len(problem_entities)}")
            print(f"  提取关系数: {len(problem_relations)}")

            # 统计题目类型实体
            problem_count = sum(1 for e in problem_entities if "题目" in e.get("entity_type_dim2", ""))
            solution_count = sum(1 for e in problem_entities if "题解" in e.get("entity_type_dim2", ""))
            trick_count = sum(1 for e in problem_entities if "技巧" in e.get("entity_type_dim2", ""))
            kp_count = sum(1 for e in problem_entities if "概念" in e.get("entity_type_dim2", ""))

            print(f"  题目实体: {problem_count}")
            print(f"  解法实体: {solution_count}")
            print(f"  技巧实体: {trick_count}")
            print(f"  知识点实体: {kp_count}")
        except Exception as e:
            print(f"题目提取失败: {e}")

    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 打印计时报告
        print("\n" + "="*60)
        print("LightRAG 新架构性能分析报告")
        print("="*60)

if __name__ == "__main__":
    # 导入LLM使用跟踪器清理函数
    from llm_tracker import cleanup

    try:
        import sys
        asyncio.run(main())
    finally:
        # 程序结束时自动打印报告并卸载tracker
        cleanup()
