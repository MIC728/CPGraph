"""
Neo4j 向量索引创建脚本（使用官方异步驱动）
创建 Entity 和 Relationship 的向量索引以支持相似度搜索
"""

import os
import sys
import asyncio
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

# 加载环境变量
# 尝试多个可能的 .env 路径
import pathlib
script_dir = pathlib.Path(__file__).parent.resolve()
project_root = script_dir.parent

# 优先从项目根目录加载，其次从脚本目录的父目录加载
for env_path in [project_root / ".env", script_dir / ".env", pathlib.Path(".env")]:
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=False)
        break


async def create_vector_indexes():
    """创建 Neo4j 向量索引"""
    print("=" * 60)
    print("Neo4j 向量索引创建工具")
    print("=" * 60)

    driver = None
    try:
        # 读取配置
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

        print(f"\n📊 配置信息:")
        print(f"  URI: {uri}")
        print(f"  用户: {user}")
        print(f"  向量维度: {embedding_dim}")

        # 创建连接
        print(f"\n🔌 连接 Neo4j...")
        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

        # 测试连接
        await driver.verify_connectivity()
        print(f"✅ 连接成功!")

        # 检查现有索引
        print(f"\n🔍 检查现有向量索引...")
        async with driver.session() as session:
            result = await session.run("SHOW INDEXES")
            existing_indexes = []
            async for record in result:
                existing_indexes.append({
                    'name': record['name'],
                    'type': record.get('type', ''),
                    'state': record.get('state', ''),
                    'populationPercent': record.get('populationPercent', '')
                })

            vector_indexes = [idx for idx in existing_indexes if idx['type'] == 'VECTOR']

            if vector_indexes:
                print(f"  现有向量索引:")
                for idx in vector_indexes:
                    print(f"    - {idx['name']} ({idx['state']})")
            else:
                print(f"  未找到向量索引")

            # 创建 Entity 向量索引
            print(f"\n📦 创建 Entity 向量索引...")
            entity_index_name = "entity_embedding_index"

            # 检查索引是否已存在，如果存在则先删除
            index_exists = any(idx['name'] == entity_index_name for idx in existing_indexes)
            if index_exists:
                print(f"  删除旧索引: {entity_index_name}")
                await session.run(f"DROP INDEX {entity_index_name} IF EXISTS")

            # 创建新索引
            await session.run(f"""
                CREATE VECTOR INDEX {entity_index_name}
                FOR (e:Entity) ON (e.embedding)
                OPTIONS {{indexConfig: {{
                  `vector.dimensions`: {embedding_dim},
                  `vector.similarity_function`: 'cosine'
                }}}}
            """)

            print(f"  ✅ Entity 向量索引创建成功: {entity_index_name}")

            # 计算并存储全图 PageRank
            print(f"\n🧮 计算全图 PageRank...")
            try:
                # 检查 GDS 是否可用
                result = await session.run("RETURN gds.version() AS version")
                record = await result.single()
                gds_version = record['version']
                print(f"  ✅ GDS 版本: {gds_version}")

                # 使用 GDS 计算 PageRank
                print(f"  📊 创建图投影...")
                await session.run("""
                    CALL gds.graph.project(
                        'entity_graph',
                        'Entity',
                        {
                            RELATIONSHIP: {
                                type: '*',
                                orientation: 'UNDIRECTED'
                            }
                        }
                    )
                """)

                print(f"  🧮 运行 PageRank 算法...")
                result = await session.run("""
                    CALL gds.pageRank.write('entity_graph', {
                        writeProperty: 'pagerank',
                        dampingFactor: 0.85,
                        maxIterations: 40
                    })
                    YIELD nodePropertiesWritten
                    RETURN nodePropertiesWritten
                """)
                record = await result.single()
                pagerank_result = record['nodePropertiesWritten']

                print(f"  ✅ PageRank 计算完成，写入了 {pagerank_result} 个节点")

                # 删除图投影以释放内存
                print(f"  🧹 清理图投影...")
                await session.run("CALL gds.graph.drop('entity_graph')")

                # 为 pagerank 创建索引以加速查询
                print(f"\n📇 创建 PageRank 索引...")
                await session.run("""
                    CREATE INDEX entity_pagerank_idx IF NOT EXISTS
                    FOR (e:Entity) ON (e.pagerank)
                """)
                print(f"  ✅ PageRank 索引创建成功")

            except Exception as e:
                error_msg = str(e)
                if "gds.version" in error_msg or "ProcedureNotFound" in error_msg:
                    print(f"  ⚠️ GDS 插件未安装或未启用")
                    print(f"     请安装并启用 GDS 插件：https://neo4j.com/docs/graph-data-science/")
                else:
                    print(f"  ⚠️ PageRank 计算失败: {e}")
                    print(f"     可能原因：权限不足或图数据问题")
                print(f"     跳过 PageRank 计算")

            # 检查 Entity 数据统计
            print(f"\n📈 Entity 数据统计:")
            result = await session.run("MATCH (e:Entity) RETURN count(e) as count")
            record = await result.single()
            total_entities = record['count']

            result = await session.run("""
                MATCH (e:Entity) WHERE e.embedding IS NOT NULL RETURN count(e) as count
            """)
            record = await result.single()
            entities_with_embedding = record['count']

            # PageRank 统计
            try:
                result = await session.run("""
                    MATCH (e:Entity) WHERE e.pagerank IS NOT NULL RETURN count(e) as count
                """)
                record = await result.single()
                entities_with_pagerank = record['count']
                print(f"  总 Entity 数量: {total_entities}")
                print(f"  包含向量的 Entity: {entities_with_embedding}")
                print(f"  向量覆盖率: {entities_with_embedding/total_entities*100:.1f}%" if total_entities > 0 else "  无数据")
                print(f"  包含 PageRank 的 Entity: {entities_with_pagerank}")
                print(f"  PageRank 覆盖率: {entities_with_pagerank/total_entities*100:.1f}%" if total_entities > 0 else "  无数据")
            except:
                print(f"  总 Entity 数量: {total_entities}")
                print(f"  包含向量的 Entity: {entities_with_embedding}")
                print(f"  向量覆盖率: {entities_with_embedding/total_entities*100:.1f}%" if total_entities > 0 else "  无数据")

            # 验证索引状态
            print(f"\n✅ 验证索引状态...")
            result = await session.run(f"""
                SHOW INDEXES WHERE name = '{entity_index_name}'
            """)
            index_records = [record async for record in result]

            if index_records:
                index_info = index_records[0]
                state = index_info.get('state', '')
                population = index_info.get('populationPercent', '')
                print(f"  索引状态: {state}")
                print(f"  构建进度: {population}%")
            else:
                print(f"  ❌ 索引未找到或创建失败")

            # 尝试测试索引（仅在有数据时）
            if entities_with_embedding > 0:
                print(f"\n🧪 测试向量索引...")
                try:
                    # 获取一个示例向量
                    result = await session.run("""
                        MATCH (e:Entity)
                        WHERE e.embedding IS NOT NULL
                        RETURN e.embedding as vector LIMIT 1
                    """)
                    record = await result.single()
                    sample = record['vector']

                    if sample:
                        test_result = []
                        result = await session.run(f"""
                            CALL db.index.vector.queryNodes(
                                '{entity_index_name}',
                                3,
                                $vector
                            ) YIELD node, score
                            RETURN node.entity_id as name, score
                        """, vector=sample)
                        async for record in result:
                            test_result.append({
                                'name': record['name'],
                                'score': record['score']
                            })

                        if test_result:
                            print(f"  ✅ 索引测试成功，返回 {len(test_result)} 个结果")
                            for row in test_result:
                                print(f"    - {row['name']}: {row['score']:.3f}")
                        else:
                            print(f"  ⚠️ 索引测试无返回结果")
                except Exception as e:
                    print(f"  ⚠️ 索引测试失败: {e}")

        print("\n" + "=" * 60)
        print("✅ 向量索引创建完成!")
        print("=" * 60)
        print("\n💡 使用方法:")
        print("  1. 向量搜索：调用 find_similar_entities() 可使用向量相似度搜索")
        print("  2. 重排序：设置 rerank='degree' 或 rerank='pagerank' 进行重排序")
        print("     - degree: 基于候选子图度数重排序")
        print("     - pagerank: 基于全图 PageRank 重排序（需要 GDS 插件）")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        print(f"\n🔧 故障排除:")
        print(f"  1. 检查 Neo4j 版本 (需要 5.x 支持 Vector Index)")
        print(f"  2. 确认数据库中有包含 embedding 字段的 Entity")
        print(f"  3. 验证向量维度设置正确")
        print(f"  4. 检查用户权限 (需要 CREATE INDEX 权限)")
        print(f"  5. PageRank 计算需要 GDS 插件，请检查是否安装并启用")
        print(f"     安装方法：在 Neo4j Desktop 中安装 GDS 插件，或参考 https://neo4j.com/docs/graph-data-science/")
        print("=" * 60)
        return False

    finally:
        if driver:
            await driver.close()


if __name__ == "__main__":
    success = asyncio.run(create_vector_indexes())
    sys.exit(0 if success else 1)