# CPGraph：编程竞赛知识图谱

一个基于 Neo4j 图数据库的信息学竞赛知识图谱系统，通过 MCP (Model Context Protocol) 提供智能查询服务。

## 📖 项目由来

最初本项目基于 LightRAG 开发，但 LightRAG 定位为轻量级 RAG 任务，文本提取的并行度不足。现通过多线程实现自主接管了文本提取和实体合并过程，目前仅复用了 LightRAG 的少量基础工具代码。

## 🌟 核心特性

- **🔌 MCP服务**：支持自定义的复杂查询

- **⚡ 多线程提取**：大幅提升文本处理效率

- **✨ 高质量实体对齐**：使用向量相似度和LLM合并子图


## 🚀 快速开始

### 环境要求

- Python 3.12+
- Neo4j 5.0+
- 至少 2GB 可用内存

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd CPGraph
```

2. **安装依赖**
```bash
uv sync
```

3. **配置环境变量**
```bash
cp .env.example .env
# 然后编辑 .env 文件，填入你的配置
```

4. **启动服务**
```bash
python mcp_server/main.py
```

## 📝 如何使用文本提取

本系统采用两步式文本提取流程：**文件提取** → **实体合并**。

### 数据准备

首先准备你的数据文件，支持两种格式：

1. **普通文档**（支持 .txt, .md, .html 等文本格式）
   ```
   documents/
   ├── algorithm_analysis.md
   ├── data_structure_guide.txt
   └── graph_theory.html
   ```

2. **竞赛题目**（JSONL 格式，每行一个题目）
   ```
   problems.jsonl
   ```
   题目格式示例：
   ```json
   {
     "title": "最大子段和",
     "difficulty": "Medium",
     "tags": ["动态规划", "数组"],
     "description": "给定一个整数数组，找到和最大的连续子数组...",
     "input_format": "第一行包含一个整数 n...",
     "output_format": "输出最大子段和",
     "constraints": "1 ≤ n ≤ 10^5",
     "sample_input": "5\n1 -2 3 4 -3",
     "sample_output": "7",
     "source": "洛谷 P1115"
   }
   ```

### 第一步：文件提取和实体生成

使用 `file_extract.py` 进行文档解析和初始实体提取：

```bash
# 提取普通文档
python file_extract.py ./documents --mode normal

# 提取竞赛题目
python file_extract.py ./problems.jsonl --mode problem
```

**输出文件：**
- `kg_storage/entities.json` - 初始实体列表
- `kg_storage/relations.json` - 初始关系列表
- `kg_storage/chunks.json` - 文本块列表

### 第二步：实体合并和优化

使用 `entity_merge.py` 进行智能实体合并和去重：

```bash
# 直接运行（使用默认参数）
python entity_merge.py
```

**参数说明：**
- `--save`: 仅保存模式，从 merged_data 读取数据并存储到 Neo4j
- `--data-dir`: 指定数据目录（默认：./merged_data）
- `--clear`: 是否清空现有数据（默认：true）
- `--no-index`: 不创建向量索引

**注意：** 确保 `.env` 文件中的 `EXTRACTOR_OUTPUT_DIR` 与 `file_extract.py` 的输出目录一致（默认为 `./kg_storage`）。

**主要功能：**
- 🔄 **增量更新**：支持多次提取，自动合并新数据
- 🎯 **双策略合并**：
  - 普通实体：使用 LLM 进行高质量语义合并
  - 题目实体：使用向量相似度快速合并
- 🏷️ **类型校正**：自动识别和修正实体类型
- ✂️ **去重优化**：消除重复实体，统一描述

**输出文件：**
- `merged_data/processed_entities.json` - 合并后的实体列表
- `merged_data/processed_relations.json` - 优化后的关系列表
- `merged_data/chunks_backup.json` - 文本块备份

### 配置选项

可以通过 `.env` 文件配置处理参数，主要选项包括：

**关键配置：**
- `EXTRACTOR_OUTPUT_DIR`: file_extract.py 输出目录（默认：./kg_storage）
- `MERGED_DATA_DIR`: entity_merge.py 输出目录（默认：./merged_data）
- `MCP_PORT`: MCP 服务器监听端口（默认：8000）
- `CLEAR_NEO4J`: 是否清空 Neo4j 现有数据（默认：true）
- `ENTITY_MERGE_INCREMENTAL`: 是否启用增量更新（默认：false）
- `MAX_WORKERS`: 并行处理线程数
- `LLM_MODEL`: LLM 模型名称
- `EMBEDDING_MODEL`: 向量模型
- `SIMILARITY_THRESHOLD`: 相似度阈值（题目合并）
- `CHUNK_SIZE`: 文本块大小
- `CHUNK_OVERLAP_SIZE`: 文本块重叠大小

详细说明请参考 `.env.example` 文件中的注释。
