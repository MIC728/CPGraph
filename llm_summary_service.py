"""
简洁的LLM摘要服务 - 直接调用OpenAI API，移除对外部llm_func的依赖
参考lightrag/operate.py、test.py和lightrag/prompt.py的设计理念
"""

import asyncio
import json
import os
from typing import List, Dict, Any, Optional, Union
from lightrag.prompt import PROMPTS
from lightrag.llm.openai import openai_complete_if_cache


class LLMSummaryService:
    """简洁的LLM摘要服务类 - 直接调用OpenAI API"""
    
    def __init__(self):
        """初始化LLM服务"""
        # 从环境变量获取配置
        self.model = os.getenv("LLM_MODEL", "deepseek-chat")
        self.api_key = os.getenv("LLM_BINDING_API_KEY")
        self.base_url = os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com")
        
        if not self.api_key:
            raise ValueError("未找到LLM_API_KEY环境变量，请检查.env配置")
    
    async def summarize_descriptions(
        self,
        description_type: str,
        description_name: str,
        description_list: List[str],
        separator: str = "\n",
        summary_length: str = "One Paragraphs",
        language: str = "zh-CN",
        **kwargs
    ) -> str:
        """
        合并多个描述为单一摘要
        
        Args:
            description_type: "Entity" 或 "Relation"
            description_name: 描述对象名称
            description_list: 描述列表
            separator: 描述之间的分隔符
            summary_length: 摘要长度
            language: 输出语言
            **kwargs: 传递给LLM函数的额外参数
            
        Returns:
            合并后的摘要文本
        """
        if not description_list:
            return ""
        
        # 少于等于4个描述时，直接拼接（参考operate.py的逻辑）
        if len(description_list) <= 4:
            unique_descriptions = list(dict.fromkeys(description_list))
            return separator.join(unique_descriptions)
        
        # 构建JSON格式的描述数据（参考operate.py的格式）
        json_descriptions = [{"Description": desc} for desc in description_list]
        joined_descriptions = separator.join(
            json.dumps(desc, ensure_ascii=False) for desc in json_descriptions
        )
        
        # 构造prompt（参考operate.py的prompt设计）
        prompt = self._build_summary_prompt(
            description_type=description_type,
            description_name=description_name,
            description_data=joined_descriptions,
            summary_length=summary_length,
            language=language
        )
        
        try:
            # 直接调用OpenAI API
            response = await openai_complete_if_cache(
                model=self.model,
                prompt=prompt,
                system_prompt="你是一个专业的文本摘要助手，专门用于合并和总结实体和关系描述。",
                api_key=self.api_key,
                base_url=self.base_url,
                enable_cot=True,
                **kwargs
            )
            
            # 清理响应文本
            if isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            # LLM失败时的fallback（参考operate.py的错误处理）
            return separator.join(description_list)
    
    def _build_summary_prompt(
        self,
        description_type: str,
        description_name: str,
        description_data: str,
        summary_length: str,
        language: str
    ) -> str:
        """
        构建摘要prompt（使用lightrag/prompt.py中的现有模板）
        """
        # 使用PROMPTS中现有的summarize_entity_descriptions模板
        prompt_template = PROMPTS["summarize_entity_descriptions"]
        
        return prompt_template.format(
            description_type=description_type,
            description_name=description_name,
            description_list=description_data,
            summary_length=summary_length,
            language=language
        )
    
    async def extract_keywords(
        self,
        text: str,
        language: str = "zh-CN",
        **kwargs
    ) -> Dict[str, List[str]]:
        """
        提取关键词（使用lightrag/prompt.py中的keywords_extraction模板）
        
        Args:
            text: 待提取关键词的文本
            language: 语言
            **kwargs: 传递给LLM函数的额外参数
            
        Returns:
            包含high_level_keywords和low_level_keywords的字典
        """
        # 使用PROMPTS中现有的keywords_extraction模板
        prompt_template = PROMPTS["keywords_extraction"]
        examples = "\n".join(PROMPTS["keywords_extraction_examples"])
        
        prompt = prompt_template.format(
            query=text,
            examples=examples
        )

        try:
            response = await openai_complete_if_cache(
                model=self.model,
                prompt=prompt,
                system_prompt="你是一个专业的关键词提取助手，擅长从文本中识别重要的概念和细节。",
                api_key=self.api_key,
                base_url=self.base_url,
                keyword_extraction=True,
                **kwargs
            )
            
            # 解析JSON响应
            if isinstance(response, str):
                import json_repair
                result = json_repair.loads(response)
            else:
                result = json.loads(str(response))
            
            return {
                "high_level_keywords": result.get("high_level_keywords", []),
                "low_level_keywords": result.get("low_level_keywords", [])
            }
            
        except Exception as e:
            # 提取失败时的fallback
            return {
                "high_level_keywords": [],
                "low_level_keywords": []
            }
    
    async def format_response(
        self,
        content: str,
        response_type: str = "Multiple Paragraphs",
        **kwargs
    ) -> str:
        """
        格式化LLM响应（参考operate.py的响应处理逻辑）
        
        Args:
            content: 原始响应内容
            response_type: 期望的响应类型
            **kwargs: 传递给LLM函数的额外参数
            
        Returns:
            格式化后的响应
        """
        if not content:
            return ""
        
        # 基本的清理逻辑（参考operate.py中的清理代码）
        cleaned_content = (
            content.replace("<system>", "")
                   .replace("</system>", "")
                   .replace("user", "")
                   .replace("model", "")
                   .strip()
        )
        
        return cleaned_content


# 便捷函数
def create_llm_service() -> LLMSummaryService:
    """
    创建LLM摘要服务的便捷函数
    
    Returns:
        LLMSummaryService实例
    """
    return LLMSummaryService()


