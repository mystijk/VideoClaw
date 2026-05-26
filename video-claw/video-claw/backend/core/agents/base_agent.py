# -*- coding: utf-8 -*-
"""
智能体基类 - 所有阶段智能体的抽象接口
"""

import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Callable
from config import settings

logger = logging.getLogger(__name__)


class AgentInterface(ABC):
    """所有智能体必须实现的接口"""

    def __init__(self, name: str = ""):
        self.name = name
        self.cancellation_check: Optional[Callable] = None
        self.progress_callback: Optional[Callable] = None

    def _merge_session_params(self, input_data: Any) -> Dict:
        """从 session.json 补齐缺失的参数"""
        if not isinstance(input_data, dict):
            return {}
            
        sid = input_data.get("session_id")
        if not sid:
            return input_data

        session_file = os.path.join(settings.SESSION_DIR, f"{sid}.json")
        if not os.path.exists(session_file):
            return input_data

        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # 基础参数列表
            keys_to_merge = [
                "style", "video_ratio", "llm_model", "vlm_model", 
                "image_t2i_model", "image_it2i_model", "video_model",
                "video_style", "expand_idea", "enable_concurrency"
            ]
            
            merged_data = input_data.copy()
            for key in keys_to_merge:
                # 只有当 input_data 中缺失该参数时，才从 session 中补齐
                if key not in merged_data or not merged_data[key]:
                    if key in session_data:
                        merged_data[key] = session_data[key]
            
            return merged_data
        except Exception as e:
            logger.error(f"Error merging session params: {e}")
            return input_data

    def set_cancellation_check(self, fn: Callable):
        self.cancellation_check = fn

    def set_progress_callback(self, fn: Callable):
        self.progress_callback = fn

    def _report_progress(self, phase: str, step_desc: str, percent: float, data: dict = None):
        if self.progress_callback:
            self.progress_callback(phase, step_desc, percent, data)

    def _check_cancel(self):
        if self.cancellation_check and self.cancellation_check():
            raise RuntimeError(f"Agent [{self.name}] cancelled by user")

    def _require_input(self, input_data: Dict, key: str) -> str:
        value = input_data.get(key)
        if not value:
            raise ValueError(f"Missing required model configuration: {key}")
        return str(value)

    def _cancellable_query(self, llm, prompt: str, image_urls=[], model="gemini-3-flash-preview", safe_content=True, task_id=None, web_search=False):
        """在 LLM 调用前后检查取消状态"""
        self._check_cancel()
        # 将位置参数映射给 llm.query
        result = llm.query(prompt, image_urls, model, safe_content, task_id, web_search)
        self._check_cancel()
        return result

    def _get_style_prompt(self, style_name: str) -> str:
        """从 prompts/style/{style_name}.txt 读取对应的视觉提示词"""
        import os
        style_file = os.path.join('prompts', 'style', f"{style_name}.txt")
        if os.path.exists(style_file):
            with open(style_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        # Fallback to English style name if file doesn't exist
        return style_name + " style"

    # -------- 抽象方法 --------

    @abstractmethod
    async def process(self, input_data: Any, intervention: Optional[Dict] = None) -> Dict:
        """
        核心处理逻辑

        Args:
            input_data: 来自上一阶段的输入数据
            intervention: 用户介入修改内容

        Returns:
            dict: { "payload": ..., "requires_intervention": bool }
        """
        pass
