import argparse
import asyncio
import glob
import json
import os
from pathlib import Path
from typing import List

from src.multi_threaded_extractor import extract_documents_async, extract_problems_async, ExtractionConfig

class FileExtractor:
    """CPGraph 文件提取器"""

    def __init__(self):
        self.supported_extensions = ['.md', '.txt', '.html', '.json', '.jsonl']
        self.encodings = ['utf-8', 'gbk']

    def extract(self, input_path: str) -> List[str]:
        """提取文件"""
        path = Path(input_path)

        if not path.exists():
            raise FileNotFoundError(f"路径不存在: {input_path}")

        if path.is_dir():
            texts = self._extract_from_folder(path)
        elif path.suffix == '.jsonl':
            texts = self._extract_from_jsonl(path)
        elif path.suffix == '.json':
            texts = self._extract_from_json(path)
        else:
            texts = self._extract_from_text_file(path)

        return texts

    def _extract_from_folder(self, folder_path: Path) -> List[str]:
        """从文件夹提取"""
        texts = []
        pattern = str(folder_path / "**" / "*")

        for file_path in glob.glob(pattern, recursive=True):
            path = Path(file_path)
            if path.is_file() and self._is_supported(path):
                content = self._read_file(path)
                if content and content.strip():
                    texts.append(content)

        return texts

    def _extract_from_jsonl(self, jsonl_path: Path) -> List[str]:
        """从JSONL提取 - 每行JSON直接dumps"""
        texts = []

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        texts.append(json.dumps(data, ensure_ascii=False))
                    except json.JSONDecodeError:
                        continue

        return texts

    def _extract_from_json(self, json_path: Path) -> List[str]:
        """从JSON文件提取"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [json.dumps(data, ensure_ascii=False)]

    def _extract_from_text_file(self, file_path: Path) -> List[str]:
        """从文本文件提取"""
        content = self._read_file(file_path)
        return [content] if content and content.strip() else []

    def _is_supported(self, path: Path) -> bool:
        return path.suffix in self.supported_extensions

    def _read_file(self, file_path: Path) -> str:
        for encoding in self.encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return ""
async def create_llm_func():
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
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

async def extract_and_extract(input_path: str, mode: str):
    print(f"🔍 提取文件: {input_path}")
    extractor = FileExtractor()
    texts = extractor.extract(input_path)
    print(f"✅ 提取完成: {len(texts)} 个文档")

    print("🤖 初始化LLM函数...")
    llm_func = await create_llm_func()

    if mode == "normal":
        config = ExtractionConfig(
            thread_count=int(os.getenv("THREAD_COUNT", "16")),
            max_concurrent_per_thread=int(os.getenv("MAX_CONCURRENT", "8")),
            chunk_token_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
            output_dir=os.getenv("EXTRACTOR_OUTPUT_DIR", "./kg_storage"),
            enable_progress_logging=True,
            log_interval=5
        )
        print(f"📝 模式: 普通文本提取")

    elif mode == "problem":
        config = ExtractionConfig(
            thread_count=int(os.getenv("THREAD_COUNT", "16")),
            max_concurrent_per_thread=int(os.getenv("MAX_CONCURRENT", "8")),
            chunk_token_size=10000000000,
            chunk_overlap_token_size=int(os.getenv("CHUNK_OVERLAP_SIZE", "100")),
            extraction_mode="problem",
            output_dir=os.getenv("EXTRACTOR_OUTPUT_DIR", "./kg_storage"),
            enable_progress_logging=True,
            log_interval=5
        )
        print(f"📝 模式: 题目文本提取")

    print("🚀 开始实体提取...")

    if mode == "normal":
        entities, relations = await extract_documents_async(
            documents=texts,
            llm_func=llm_func,
            config=config
        )
    else:
        entities, relations = await extract_problems_async(
            documents=texts[:10],
            llm_func=llm_func,
            config=config
        )

    print(f"\n✅ 实体提取完成!")
    print(f"📊 统计信息:")
    print(f"   - 文档数量: {len(texts)}")
    print(f"   - 提取实体: {len(entities)}")
    print(f"   - 提取关系: {len(relations)}")
    print(f"   - 输出目录: {config.output_dir}")

    if mode == "problem":
        problem_count = sum(1 for e in entities if "题目" in e.get("entity_type_dim2", ""))
        solution_count = sum(1 for e in entities if "题解" in e.get("entity_type_dim2", ""))
        trick_count = sum(1 for e in entities if "技巧" in e.get("entity_type_dim2", ""))
        kp_count = sum(1 for e in entities if "概念" in e.get("entity_type_dim2", ""))

        print(f"   - 题目实体: {problem_count}")
        print(f"   - 解法实体: {solution_count}")
        print(f"   - 技巧实体: {trick_count}")
        print(f"   - 知识点实体: {kp_count}")

async def main():
    parser = argparse.ArgumentParser(
        description="CPGraph 文件提取+实体提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python file_extract.py ./documents --mode normal
  python file_extract.py ./problems.jsonl --mode problem

提取模式:
  normal   - 普通文本实体提取
  problem  - 题目专用实体提取（不分割题目和题解）
        """
    )

    parser.add_argument('input_path', help='输入路径（文件夹或文件）')
    parser.add_argument('--mode', choices=['normal', 'problem'], default='normal',
                       help='提取模式: normal(普通文本) 或 problem(题目文本)，默认 normal')

    args = parser.parse_args()

    try:
        await extract_and_extract(args.input_path, args.mode)
        return 0
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    from src.llm_tracker import cleanup

    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    finally:
        cleanup()