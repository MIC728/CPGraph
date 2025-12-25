from __future__ import annotations
from typing import Any

PROMPTS: dict[str, Any] = {}

# All delimiters must be formatted as "<|UPPER_CASE_STRING|>"
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = "<|#|>"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"

# ==================== 题目专用实体提取 Prompt ====================

PROMPTS["problem_entity_extraction_system_prompt"] = """
---Role---
你是一名信息学竞赛专家，负责从竞赛题目题解中提取与题目相关的实体和关系，并为题目实体生成高度精炼的语义摘要。

---Instructions---
1.  **严格类型约束与实体提取:**
    *   **⚠️ 重要：类型使用规范（必须严格遵守）**
        *   **允许的第一维度类型（entity_type_dim1）：**
            - `数据结构`：用于存储和组织数据的方式（如线段树、并查集、树状数组）
            - `算法思想`：通用的算法设计思想（如分治、贪心、回溯）
            - `动态规划`：动态规划相关概念和优化技巧
            - `图论`：图结构、图算法和图相关概念
            - `搜索`：搜索算法和搜索策略
            - `字符串`：字符串处理算法和数据结构
            - `数学`：数学定理、公式、数论概念
            - `计算几何`：几何算法和几何数据结构
            - `其他`：仅当以上类型都无法适用时使用

        *   **允许的第二维度类型（entity_type_dim2）：**
            - `概念`：抽象的理论概念和定义
            - `技巧`：具体的解题技巧和优化方法
            - `实现`：算法的具体实现方式或数据结构的具体操作
            - `模型`：数学模型或问题抽象模型
            - `算法`：完整的算法或算法步骤
            - `原理`：算法或方法的理论基础和原理（**包含定理、定律、公式等理论**）
            - `题目`：完整的竞赛题目
            - `其他`：仅当以上类型都无法适用时使用

        *   **🚫 严格禁止的类型：**
            - **通用无效标签**：Entity、Object、Item、Thing、Element、Unit
            - **动词或动作词**：运算、操作、处理、分析、计算
            - **语言标识符**：English、Chinese、CN、EN
            - **元概念**：元数据、标签、属性、字段
            - **常见幻觉类型**：定理、理论、方法、技术、系统
            - **任何不在上述允许列表中的自定义类型**

        *   **类型选择原则：**
            - 必须从上述允许的类型中选择，不得创造新类型
            - 优先选择最具体和最匹配的类型
            - 当实在无法确定时，选择"其他"而不是创造新类型
            - 同一实体的两个维度类型必须严格从对应列表中选择
            - 题目类型判断时，重点关注题目名称和描述中的关键词仔细提取类型（如"平衡树"、"线段树"等直接对应数据结构类型），而不是全填“其他”

    *   **识别范围：** 识别题目的题意，核心知识点和重要技巧。严格限制提取范围，只提取以下类型的实体（此规则优先级高于一切）：
        - 经典算法和数据结构名称
        - 竞赛中的核心概念和技巧名称
        - 重要定理和公式
        - 竞赛中的通用解题思路和方法
        - 解题中出现的Trick，解题需要的核心知识点
        - **题目相关实体**

    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: 使用标准名称，保持一致性；保留原始专有名词
        *   `entity_type_dim1`: 第一维度技术分类（只允许选择一个！），优先使用竞赛相关类型：{entity_types_dim1}。若无适用类型，则使用“其他”
        *   `entity_type_dim2`: 第二维度应用层次（可多选），优先使用竞赛相关类型：{entity_types_dim2}。若无适用类型，则使用“其他”
        *   `entity_description`: 简洁描述，突出核心概念和竞赛应用场景，**避免具体实现细节和使用场景**。
            *   **特别地，对于`entity_type_dim2`为“题目”的实体**：其`entity_description`**必须**是一个专门生成的语义摘要（生成方法见第3部分）。

    *   **Output Format - Entities:** Output a total of 5 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type_dim1{tuple_delimiter}entity_type_dim2{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Relationship Schema:** 严格使用以下11个标准关系类型中的一个或多个（用逗号分隔）：
        *   **IS_A**: 表达严格的分类关系（X是Y的一种/一类/实例，不包含定义关系）
        *   **PART_OF**: 表达整体与部分的组成关系（包含、构成、组成、分解）
        *   **BASED_ON**: 表达知识依赖或前提条件（基于、依赖、前提、源于、原理）
        *   **APPLIES_TO**: 表达通用方法用于解决具体问题（应用、应用于、实现、解决、处理）
        *   **EVALUATES**: 表达评估、验证、测试（正确性、性能、效果验证）
        *   **EXPLAINS**: 表达分析、解释、阐明（算法性质、问题结构、理论原理）
        *   **PRACTICED_BY**: 表达知识被题目练习/测试（知识应用于具体题目场景）
        *   **COMPARES_WITH**: 表达对比、关联或类比（对比、关联、类似、等价、替代）
        *   **LEADS_TO**: 表达逻辑推导、衍生或因果（推导、转化、导致、生成、结论）
        *   **OPTIMIZES**: 表达在特定维度上的改进（优化、简化、加速、改进）
        *   **TRANSFORMS_TO**: 表达形式间的系统性转换（转化、转换、映射、模型转换）

    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction.
        *   `relationship_keywords`: 使用标准关系类型（可多个，用逗号分隔，如：APPLIES_TO,BASED_ON）
        *   `relationship_description`: 简洁描述，强调实体间的可泛化的逻辑关系。

    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **题目专用提取规则：**
    *   **最重要规则**：对于每个文本，只能提取一个题目类型实体，简述题意，技巧和知识点个数不限。题目与使用到的关键技巧（如果有）用关系连接，技巧和与技巧相关的核心知识点用关系连接。
          也就是：题目--[某种关系]--技巧（可能没有）--[某种关系]--核心知识点
    *   **题目实体提取**：
        *   提取题目的核心描述和约束条件，使用`entity_type_dim2=“题目”`，entity_type_dim1根据题意从允许的第一维度类型（entity_type_dim1）标签中选择（只允许选择一个！），`entity_name`必须使用题目ID开头，且题目ID末尾用空格分隔。
        *   **题目实体的`entity_description`字段生成规则**：你**必须**为每个题目实体生成一个专门的语义摘要。请基于**题目描述**和**题解思路**，完成以下分析并整合成摘要：
            1.  **抽象题意**：抛开所有故事背景，题目可以抽象为什么纯粹的数学模型或数据结构？
            2.  **核心挑战**：本题最独特、最关键的难点或约束是什么？
            3.  **关键操作**：解决上述挑战，最核心的算法操作或转换步骤是什么？
            *   **最终摘要句式**：将以上分析整合成一个**流畅、自然、信息密集的段落**，句式结构为：“该问题可抽象为`[抽象题意]`，其核心挑战在于`[核心挑战]`。解决的关键在于`[关键操作]`。”
            *   **写作要求**：
                - 直接填充：用分析的结果直接替换上述句式中的 `[ ]` 部分，保持语言精炼，总结正确。
                - 内容具体：在`[抽象题意]`和`[关键操作]`中，请使用明确的算法术语（如“基环内向森林”、“二分答案+最短路验证”）。
                - 题意应当高度形式化，不要保留任何多余的故事叙述和人名，不要描述程序的输入输出格式。题意提取可以使用latex。
    *   **技巧实体提取**：提取具体的解题技巧和优化方法，使用`entity_type_dim2=“技巧”`，技巧命名尽量使用标准术语或者易懂无歧义的命名，也可以直接根据题目名命名。
    *   **知识点实体推断提取**：
        - 必须提取题解中使用的所有算法、数据结构、理论概念
        - 即使题解没有明确定义这些概念，也要提取（如“线段树”、“状压DP”等）
        - 避免提取过于宽泛的知识点，确保这些知识点与题目密切相关，范围恰到好处。（如提取“斜率优化DP”，而不是宽泛的“动态规划”）
        - 根据具体内容确定`entity_type_dim1`（如“数据结构”、“算法思想”等）
        - 确保知识点的专业术语准确性

4.  **严格过滤规则（必须遵守）：**
    *   **保留规则：** 1. 核心算法和数据结构 2. 重要概念和解题技巧 3. 经典组合和应用模式 4. 题目和解法实体
    *   **严格排除：** 1. 具体函数名和方法名 2. 内部实现细节 3. 题目特定的技术细节 4. 过于具体的参数和变量 5. 底层操作细节（如pushup、pushdown等）

5.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

6.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

7.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.

8.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.

9.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.

10. **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types_dim1: {entity_types_dim1}
Entity_types_dim2: {entity_types_dim2}
Text:
```
{input_text}
```
"""

PROMPTS["entity_extraction_system_prompt"] = """---Role---
你是一名知识图谱专家，负责从信息学竞赛题解中提取实体和关系。核心目标是提取信息学竞赛中的核心概念和算法，去除过于详细的技术实现细节。

---Instructions---
1.  **严格类型约束与实体提取:**
    *   **⚠️ 重要：类型使用规范（必须严格遵守）**
        *   **允许的第一维度类型（entity_type_dim1）：**
            - `数据结构`：用于存储和组织数据的方式（如线段树、并查集、树状数组）
            - `算法思想`：通用的算法设计思想（如分治、贪心、回溯）
            - `动态规划`：动态规划相关概念和优化技巧
            - `图论`：图结构、图算法和图相关概念
            - `搜索`：搜索算法和搜索策略
            - `字符串`：字符串处理算法和数据结构
            - `数学`：数学定理、公式、数论概念
            - `计算几何`：几何算法和几何数据结构
            - `其他`：仅当以上类型都无法适用时使用

        *   **允许的第二维度类型（entity_type_dim2）：**
            - `概念`：抽象的理论概念和定义
            - `技巧`：具体的解题技巧和优化方法
            - `实现`：算法的具体实现方式或数据结构的具体操作
            - `模型`：数学模型或问题抽象模型
            - `算法`：完整的算法或算法步骤
            - `原理`：算法或方法的理论基础和原理（**包含定理、定律、公式等理论**）
            - `题目`：完整的竞赛题目
            - `其他`：仅当以上类型都无法适用时使用

        *   **🚫 严格禁止的类型：**
            - **通用无效标签**：Entity、Object、Item、Thing、Element、Unit
            - **动词或动作词**：运算、操作、处理、分析、计算
            - **语言标识符**：English、Chinese、CN、EN
            - **元概念**：元数据、标签、属性、字段
            - **常见幻觉类型**：定理、理论、方法、技术、系统
            - **任何不在上述允许列表中的自定义类型**

        *   **类型选择原则：**
            - 必须从上述允许的类型中选择，不得创造新类型
            - 优先选择最具体和最匹配的类型
            - 当无法确定时，选择"其他"而不是创造新类型
            - 同一实体的两个维度类型必须严格从对应列表中选择
            - 题目类型判断时，重点关注题目名称和描述中的关键词（如"平衡树"、"线段树"等直接对应数据结构类型）
            - entity_type_dim1只允许选择一个类型！
            - entity_type_dim2支持多个类型，用逗号分隔保持相对顺序（如：数据结构,算法思想）

    *   **识别范围：** 识别信息学竞赛中的核心概念、算法、数据结构和重要技巧。**严格限制提取范围，只提取以下类型的实体（此规则优先级高于一切）！**：
        - 经典算法和数据结构名称
        - 竞赛中的核心概念和技巧名称
        - 重要定理和公式
        - 竞赛中的通用解题思路和方法
        - 冷门但有用的Trick
    *   **排除以下过于详细的内容：**
        - 具体的函数名（如：pushup, pushdown, rotate等）
        - 算法的具体实现细节
        - 数据结构的内部操作细节
        - 题目特定的技术参数和变量
    *   **Entity Details:** For each identified entity, extract the following information:
        *   `entity_name`: 使用标准名称，保持一致性；保留原始专有名词。
        *   `entity_type_dim1`: 第一维度技术分类，优先使用竞赛相关类型：{entity_types_dim1}。若无适用类型，则使用"其他"。
        *   `entity_type_dim2`: 第二维度应用层次，优先使用竞赛相关类型：{entity_types_dim2}。若无适用类型，则使用"其他"。**支持多个类型，用逗号分隔保持相对顺序（如：技巧,模型,算法）**。
        *   `entity_description`: 简洁描述，突出核心概念和竞赛应用场景，**避免具体实现细节和使用场景**。注意：实体描述只关注这个实体本身信息。
    *   **Output Format - Entities:** Output a total of 5 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
        *   Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type_dim1{tuple_delimiter}entity_type_dim2{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** 识别核心实体之间重要的、可泛化的关系。重点关注算法与数据结构的组合、常见解题模式、重要概念、有用Trick间的逻辑关联。**排除过于具体的实现关联和题目特定的依赖关系**。
    *   **N-ary Relationship Decomposition:** If a single statement describes a relationship involving more than two entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
        *   **Example:** For "Alice, Bob, and Carol collaborated on Project X," extract binary relationships such as "Alice collaborated with Project X," "Bob collaborated with Project X," and "Carol collaborated with Project X," or "Alice collaborated with Bob," based on the most reasonable binary interpretations.
    *   **Relationship Schema:** 严格使用以下11个标准关系类型中的一个或多个（用逗号分隔）：
        *   **IS_A**: 表达严格的分类关系（X是Y的一种/一类/实例，不包含定义关系）
        *   **PART_OF**: 表达整体与部分的组成关系（包含、构成、组成、分解）
        *   **BASED_ON**: 表达知识依赖或前提条件（基于、依赖、前提、源于、原理）
        *   **APPLIES_TO**: 表达通用方法用于解决具体问题（应用、应用于、实现、解决、处理）
        *   **EVALUATES**: 表达评估、验证、测试（正确性、性能、效果验证）
        *   **EXPLAINS**: 表达分析、解释、阐明（算法性质、问题结构、理论原理）
        *   **PRACTICED_BY**: 表达知识被题目练习/测试（知识应用于具体题目场景）
        *   **COMPARES_WITH**: 表达对比、关联或类比（对比、关联、类似、等价、替代）
        *   **LEADS_TO**: 表达逻辑推导、衍生或因果（推导、转化、导致、生成、结论）
        *   **OPTIMIZES**: 表达在特定维度上的改进（优化、简化、加速、改进）
        *   **TRANSFORMS_TO**: 表达形式间的系统性转换（转化、转换、映射、模型转换）
    *   **Relationship Details:** For each binary relationship, extract the following fields:
        *   `source_entity`: The name of the source entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `target_entity`: The name of the target entity. Ensure **consistent naming** with entity extraction. Capitalize the first letter of each significant word (title case) if the name is case-insensitive.
        *   `relationship_keywords`: 使用标准关系类型（可多个，用逗号分隔，如：APPLIES_TO,BASED_ON）
        *   `relationship_description`: 简洁描述，强调实体间的可泛化的逻辑关系。
    *   **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
        *   Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`

3.  **严格过滤规则（必须遵守）：**
    *   **保留规则：** 1. 核心算法和数据结构 2. 重要概念和解题技巧 3. 经典组合和应用模式
    *   **严格排除：** 1. 具体函数名和方法名 2. 内部实现细节 3. 题目特定的技术细节 4. 过于具体的参数和变量 5. 底层操作细节（如pushup、pushdown等）

4.  **Delimiter Usage Protocol:**
    *   The `{tuple_delimiter}` is a complete, atomic marker and **must not be filled with content**. It serves strictly as a field separator.
    *   **Incorrect Example:** `entity{tuple_delimiter}Tokyo<|location|>Tokyo is the capital of Japan.`
    *   **Correct Example:** `entity{tuple_delimiter}Tokyo{tuple_delimiter}location{tuple_delimiter}Tokyo is the capital of Japan.`

5.  **Relationship Direction & Duplication:**
    *   Treat all relationships as **undirected** unless explicitly stated otherwise. Swapping the source and target entities for an undirected relationship does not constitute a new relationship.
    *   Avoid outputting duplicate relationships.

6.  **Output Order & Prioritization:**
    *   Output all extracted entities first, followed by all extracted relationships.
    *   Within the list of relationships, prioritize and output those relationships that are **most significant** to the core meaning of the input text first.
    *   优先输出核心竞赛知识的关系（例如：技巧 ↔ 算法）。

7.  **Context & Objectivity:**
    *   Ensure all entity names and descriptions are written in the **third person**.
    *   Explicitly name the subject or object; **avoid using pronouns** such as `this article`, `this paper`, `our company`, `I`, `you`, and `he/she`.
    *   使用第三人称，无代词，明确命名。

8.  **Language & Proper Nouns:**
    *   The entire output (entity names, keywords, and descriptions) must be written in `{language}`.
    *   Proper nouns (e.g., personal names, place names, organization names) should be retained in their original language if a proper, widely accepted translation is not available or would cause ambiguity.
    *   若翻译可能引起歧义，则保留原始专有名词。

9.  **Completion Signal:** Output the literal string `{completion_delimiter}` only after all entities and relationships, following all criteria, have been completely extracted and outputted.

---Examples---
{examples}

---Real Data to be Processed---
<Input>
Entity_types_dim1: {entity_types_dim1}
Entity_types_dim2: {entity_types_dim2}
Text:
```
{input_text}
```
"""

PROMPTS["problem_entity_extraction_user_prompt"] = """
---Task---
从输入的竞赛题解文本中提取与题目相关的实体和关系。

---Instructions---
1.  **严格遵循格式**：严格按照系统prompt中规定的实体和关系列表格式要求，包括输出顺序、字段分隔符等
2.  **仅输出内容**：仅输出提取的实体和关系列表，不要包含任何解释性文字
3.  **完成信号**：在所有相关实体和关系提取完成后，输出 `{completion_delimiter}` 作为最后一行
4.  **输出语言**：确保输出语言为 {language}

<Output>
"""

PROMPTS["problem_entity_extraction_examples"] = [
    """<Input Text>
```
给定一个长度为n的数组a，我们需要对该数组进行区间修改和区间查询操作。具体来说，我们需要支持以下两种操作：
1. 将区间[l, r]内的所有元素增加x
2. 查询区间[l, r]内所有元素的和

我们可以使用线段树来解决这个问题。线段树是一种二叉树结构，每个节点维护一个区间内的信息。

具体实现：
1. 建树：O(n)时间复杂度
2. 区间修改：O(log n)时间复杂度
3. 区间查询：O(log n)时间复杂度

在实现时，我们需要注意懒标记的使用。当我们需要对某个区间进行修改时，如果该区间完全覆盖了当前节点对应的区间，我们就直接更新该节点的值，并标记一个懒标记，表示该节点的子节点需要后续更新。

在查询时，我们需要向下传递懒标记，确保查询结果的正确性。

除了线段树，我们还可以使用树状数组(Fenwick Tree)来解决同样的问题。树状数组的代码实现更加简洁，时间复杂度同样是O(log n)，但是不支持区间修改操作。
```

<Output>
entity<|#|>区间修改查询问题<|#|>其他<|#|>题目<|#|>需要支持区间修改和区间查询的数组操作问题，核心在于高效地处理区间更新和区间求和操作。
entity<|#|>线段树解法<|#|>数据结构<|#|>题解<|#|>使用二叉树结构的线段树来解决区间修改查询问题，通过节点维护区间信息和懒标记机制实现高效的区间操作。
entity<|#|>树状数组解法<|#|>数据结构<|#|>题解<|#|>使用Fenwick Tree来解决区间查询问题，代码简洁但仅支持点修改操作，时间复杂度O(log n)。
entity<|#|>懒标记技巧<|#|>其他<|#|>技巧<|#|>在线段树中使用的延迟更新技巧，通过标记节点表示子节点需要后续更新，优化区间修改操作的时间复杂度。
entity<|#|>线段树<|#|>数据结构<|#|>概念<|#|>一种二叉树数据结构，用于高效地处理区间修改和区间查询操作，支持O(log n)时间复杂度的区间操作。
entity<|#|>树状数组<|#|>数据结构<|#|>概念<|#|>一种支持点修改和区间查询的树形数据结构，代码简洁，时间复杂度O(log n)。
relation<|#|>线段树解法<|#|>线段树<|#|>BASED_ON<|#|>线段树解法基于线段树数据结构实现。
relation<|#|>线段树解法<|#|>懒标记技巧<|#|>APPLIES_TO<|#|>线段树解法使用懒标记技巧来优化性能。
relation<|#|>树状数组解法<|#|>树状数组<|#|>BASED_ON<|#|>树状数组解法基于树状数组数据结构实现。
relation<|#|>线段树<|#|>树状数组<|#|>COMPARES_WITH<|#|>两种都是高效的区间操作数据结构，但适用场景有所不同。
{completion_delimiter}

""",
]

PROMPTS["entity_extraction_user_prompt"] = """---Task---
Extract entities and relationships from the input text to be processed.

---Instructions---
1.  **Strict Adherence to Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system prompt.
2.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant entities and relationships have been extracted and presented.
4.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_continue_extraction_user_prompt"] = """---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted** entities and relationships from the input text.

---Instructions---
1.  **Strict Adherence to System Format:** Strictly adhere to all format requirements for entity and relationship lists, including output order, field delimiters, and proper noun handling, as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully** extracted in the last task.
    *   If an entity or relationship was **missed** in the last task, extract and output it now according to the system format.
    *   If an entity or relationship was **truncated, had missing fields, or was otherwise incorrectly formatted** in the last task, re-output the *corrected and complete* version in the specified format.
3.  **Output Format - Entities:** Output a total of 5 fields for each entity, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format - Relationships:** Output a total of 5 fields for each relationship, delimited by `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** Output *only* the extracted list of entities and relationships. Do not include any introductory or concluding remarks, explanations, or additional text before or after the list.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line after all relevant missing or corrected entities and relationships have been extracted and presented.
7.  **Output Language:** Ensure the output language is {language}. Proper nouns (e.g., personal names, place names, organization names) must be kept in their original language and not translated.

<Output>
"""

PROMPTS["entity_extraction_examples"] = [
    """<Input Text>
```
Alex observed Taylor's authoritarian behavior while Jordan showed reverence for a mysterious device. This created tension with Cruz's control-focused vision.
```

<Output>
entity{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}其他{tuple_delimiter}Alex observes group dynamics and tensions.
entity{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}其他{tuple_delimiter}Taylor exhibits authoritarian behavior and shows device reverence.
entity{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}其他{tuple_delimiter}Jordan shows reverence for a mysterious device.
entity{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}其他{tuple_delimiter}Cruz represents control-focused vision creating group tension.
entity{tuple_delimiter}The Device{tuple_delimiter}equipment{tuple_delimiter}其他{tuple_delimiter}Mysterious device with significant importance to the group.
relation{tuple_delimiter}Alex{tuple_delimiter}Taylor{tuple_delimiter}COMPARES_WITH{tuple_delimiter}Alex observes Taylor's authoritarian behavior changes.
relation{tuple_delimiter}Taylor{tuple_delimiter}Jordan{tuple_delimiter}COMPARES_WITH{tuple_delimiter}Taylor and Jordan interact regarding the device.
relation{tuple_delimiter}Jordan{tuple_delimiter}Cruz{tuple_delimiter}COMPARES_WITH{tuple_delimiter}Jordan's approach contrasts with Cruz's control vision.
{completion_delimiter}

""",
    """<Input Text>
```
Global Tech Index dropped 3.4% today as Nexon Technologies fell 7.8%. Gold futures rose 1.5% while crude oil reached $87.60 per barrel.
```

<Output>
entity{tuple_delimiter}Global Tech Index{tuple_delimiter}其他{tuple_delimiter}其他{tuple_delimiter}Tech stock index declined 3.4% today.
entity{tuple_delimiter}Nexon Technologies{tuple_delimiter}其他{tuple_delimiter}其他{tuple_delimiter}Tech company stock fell 7.8%.
entity{tuple_delimiter}Gold Futures{tuple_delimiter}其他{tuple_delimiter}其他{tuple_delimiter}Gold prices rose 1.5% as safe-haven asset.
entity{tuple_delimiter}Crude Oil{tuple_delimiter}其他{tuple_delimiter}其他{tuple_delimiter}Oil prices reached $87.60 per barrel.
relation{tuple_delimiter}Nexon Technologies{tuple_delimiter}Global Tech Index{tuple_delimiter}PART_OF{tuple_delimiter}Nexon contributed to tech index decline.
relation{tuple_delimiter}Gold Futures{tuple_delimiter}Crude Oil{tuple_delimiter}COMPARES_WITH{tuple_delimiter}Both are commodities with opposite price movements.
{completion_delimiter}

""",
]

PROMPTS["summarize_entity_descriptions"] = """---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Synthesize a list of entity/relation descriptions into a single comprehensive summary.

---Instructions---
1. Input: JSON description list with one object per line
2. Output: Plain text summary without formatting, third-person perspective
3. Comprehensiveness: Integrate all key information from every description
4. Context: Explicitly mention entity/relation name at the beginning
5. Conflict Handling: If multiple entities share same name, summarize separately
6. Length: Must not exceed {summary_length} tokens
7. Language: Write in {language}, keep proper nouns original

---Input---
{description_type} Name: {description_name}

Description List:
{description_list}

---Output---
"""

PROMPTS["fail_response"] = (
    "Sorry, I'm not able to provide an answer to that question.[no-context]"
)

PROMPTS["rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Knowledge Graph and Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize both `Knowledge Graph Data` and `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a references section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{context_data}
"""

PROMPTS["naive_rag_response"] = """---Role---

You are an expert AI assistant specializing in synthesizing information from a provided knowledge base. Your primary function is to answer user queries accurately by ONLY using the information within the provided **Context**.

---Goal---

Generate a comprehensive, well-structured answer to the user query.
The answer must integrate relevant facts from the Document Chunks found in the **Context**.
Consider the conversation history if provided to maintain conversational flow and avoid repeating information.

---Instructions---

1. Step-by-Step Instruction:
  - Carefully determine the user's query intent in the context of the conversation history to fully understand the user's information need.
  - Scrutinize `Document Chunks` in the **Context**. Identify and extract all pieces of information that are directly relevant to answering the user query.
  - Weave the extracted facts into a coherent and logical response. Your own knowledge must ONLY be used to formulate fluent sentences and connect ideas, NOT to introduce any external information.
  - Track the reference_id of the document chunk which directly support the facts presented in the response. Correlate reference_id with the entries in the `Reference Document List` to generate the appropriate citations.
  - Generate a **References** section at the end of the response. Each reference document must directly support the facts presented in the response.
  - Do not generate anything after the reference section.

2. Content & Grounding:
  - Strictly adhere to the provided context from the **Context**; DO NOT invent, assume, or infer any information not explicitly stated.
  - If the answer cannot be found in the **Context**, state that you do not have enough information to answer. Do not attempt to guess.

3. Formatting & Language:
  - The response MUST be in the same language as the user query.
  - The response MUST utilize Markdown formatting for enhanced clarity and structure (e.g., headings, bold text, bullet points).
  - The response should be presented in {response_type}.

4. References Section Format:
  - The References section should be under heading: `### References`
  - Reference list entries should adhere to the format: `* [n] Document Title`. Do not include a caret (`^`) after opening square bracket (`[`).
  - The Document Title in the citation must retain its original language.
  - Output each citation on an individual line
  - Provide maximum of 5 most relevant citations.
  - Do not generate footnotes section or any comment, summary, or explanation after the references.

5. Reference Section Example:
```
### References

- [1] Document Title One
- [2] Document Title Two
- [3] Document Title Three
```

6. Additional Instructions: {user_prompt}


---Context---

{content_data}
"""

PROMPTS["kg_query_context"] = """
Knowledge Graph Data (Entity):

```json
{entities_str}
```

Knowledge Graph Data (Relationship):

```json
{relations_str}
```

Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["naive_query_context"] = """
Document Chunks (Each entry has a reference_id refer to the `Reference Document List`):

```json
{text_chunks_str}
```

Reference Document List (Each entry starts with a [reference_id] that corresponds to entries in the Document Chunks):

```
{reference_list_str}
```

"""

PROMPTS["keywords_extraction"] = """---Role---
You are an expert keyword extractor, specializing in analyzing user queries for a Retrieval-Augmented Generation (RAG) system. Your purpose is to identify both high-level and low-level keywords in the user's query that will be used for effective document retrieval.

---Goal---
Given a user query, your task is to extract two distinct types of keywords:
1. **high_level_keywords**: for overarching concepts or themes, capturing user's core intent, the subject area, or the type of question being asked.
2. **low_level_keywords**: for specific entities or details, identifying the specific entities, proper nouns, technical jargon, product names, or concrete items.

---Instructions & Constraints---
1. **Output Format**: Your output MUST be a valid JSON object and nothing else. Do not include any explanatory text, markdown code fences (like ```json), or any other text before or after the JSON. It will be parsed directly by a JSON parser.
2. **Source of Truth**: All keywords must be explicitly derived from the user query, with both high-level and low-level keyword categories are required to contain content.
3. **Concise & Meaningful**: Keywords should be concise words or meaningful phrases. Prioritize multi-word phrases when they represent a single concept. For example, from "latest financial report of Apple Inc.", you should extract "latest financial report" and "Apple Inc." rather than "latest", "financial", "report", and "Apple".
4. **Handle Edge Cases**: For queries that are too simple, vague, or nonsensical (e.g., "hello", "ok", "asdfghjkl"), you must return a JSON object with empty lists for both keyword types.

---Examples---
{examples}

---Real Data---
User Query: {query}

---Output---
Output:"""

PROMPTS["keywords_extraction_examples"] = [
    """Query: "How does international trade influence global economic stability?"

Output:
{
  "high_level_keywords": ["International trade", "Global economic stability"],
  "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange"]
}

""",
    """Query: "What are the environmental consequences of deforestation on biodiversity?"

Output:
{
  "high_level_keywords": ["Environmental consequences", "Deforestation"],
  "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions"]
}

""",
]

# ==================== 实体合并评估 Prompt ====================

PROMPTS["entity_merge_evaluation"] = """---Role---
你是信息学竞赛知识图谱专家，负责评估实体质量并严格判断实体合并关系。

---Task---
评估当前实体是否符合质量要求，判断是否应该与已有实体合并。
**注意：只做删除和合并决策，不修改名称和类型。**

---Quality Standards---
**保留**：经典算法、数据结构、竞赛概念技巧、重要定理模型、通用解题思路、实用Trick、C++STL、完整题解。
**删除**：具体函数名、实现细节、题目特定参数、无意义概念、单个字母标识符。

---Merge Standards---
**合并**：
- 语义严格完全相同的实体（如"快速排序"和"快排"）
- 同一概念的不同表述（如"二分查找"和"二分搜索"）
- 拼写变体
- 相同题目ID的不同表述

**不合并**：
- 相关但独立的概念（如"DFS"和"BFS"）
- 不同层次的概念（如"图论"和"最短路径算法"）
- 明确不同的变体（如"Bellman-Ford算法"和"SPFA"）
- 算法与其优化版本（如"冒泡排序"和"鸡尾酒排序"）
- 题目ID不同或不同题目的实体
- 基础概念与其衍生概念（如"排序"和"快速排序"）

---Input---
当前实体：
```json
{current_entity}
```

相似实体（已截断过长的描述）：
```json
{similar_entities}
```

---Output---
```json
{{
  "should_delete": false,
  "should_merge": false,
  "merge_target": null
}}
```
"""

PROMPTS["entity_group_merge"] = """---Role---
信息学竞赛知识图谱专家，负责将多个相关实体合并为高质量节点。

---Task---
合并实体组，生成最佳名称、类型和综合描述。

---Merge Guidelines---
1. **名称选择**：选择标准通用名称，优先中文，英文通用名称（如Dijkstra）应该保留，极其通用的缩写或别称（如DFS，CDQ分治，莫队）应该保留
2. **类型确定**：选择概括性类型，横跨多类型用逗号分隔，禁止UNKNOWN，其中dim1只能选择一个类型，dim2最多选择3个类型
3. **描述合成**：提取核心要点，突出概念和应用，去除重复，不超过500字

---Entity Types---
第一维度：{entity_types_dim1}
第二维度：{entity_types_dim2}

---Input---
待合并实体：
```json
{entities_list}
```

---Output---
```json
{{
  "final_name": "合并后的实体名称",
  "final_type_dim1": "技术分类",
  "final_type_dim2": "应用层次",
  "final_description": "综合描述"
}}
```
"""

PROMPTS["entity_reclassify"] = """---Role---
信息学竞赛知识图谱专家，负责对类型不明确的实体进行分类和清洗。
注意：如果实体类型不在标准列表中，可能是因为LLM幻觉或输出错误，不要直接删除，要尽量清洗和修正信息，保留有价值的内容。
---Task---
修正实体名称和类型，清洗描述，删除无效实体。
禁止删除带题目ID的实体！
**清洗指导**：
1. 如果原类型包含幻觉或错误信息，请根据实体名称和描述推断正确类型
2. 去除多余符号和不规范表述
3. 保持实体名称的一致性
4. 描述要简洁准确，突出核心概念
---Entity Types---
第一维度（仅选一个）：{entity_types_dim1}
第二维度（可多选，用逗号分隔）：{entity_types_dim2}

---Input---
实体名称：{entity_name}
实体描述：{entity_description}

---Output---
```json
{{
  "should_delete": false,
  "corrected_name": "修正后的实体名称",
  "type_dim1": "技术分类",
  "type_dim2": "应用层次，多标签逗号分隔",
  "cleaned_description": "清洗后的实体描述"
}}
```
"""
