"""
LLM使用跟踪器 - 共享模块
为 entity_merge.py 和 new_test.py 提供统一的LLM使用统计功能
"""
import os
import functools
import time
import inspect
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime

class LLMUsageTracker:
    """LLM使用情况跟踪器"""

    def __init__(self,
                 count_tokens: bool = True,
                 count_chars: bool = True,
                 track_timing: bool = True,
                 export_to_file: bool = False,
                 export_path: str = "./llm_usage_stats.json"):
        # 统计开关
        self.count_tokens = count_tokens
        self.count_chars = count_chars
        self.track_timing = track_timing
        self.export_to_file = export_to_file
        self.export_path = export_path

        # 统计数据结构
        self.stats = {
            "session_start": datetime.now().isoformat(),
            "total_calls": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_chars_input": 0,
            "total_chars_output": 0,
            "total_time": 0.0,
            "calls_by_model": {},
            "calls_by_file": {},
            "calls_by_function": {},
            "errors": [],
            "timeline": []
        }

        # 原始函数引用（用于恢复）
        self._original_func = None
        self._is_installed = False
        self._module_path = "lightrag.llm.openai"
        self._func_name = "openai_complete_if_cache"

    def install(self) -> None:
        """安装tracker，替换原函数"""
        if self._is_installed:
            print("⚠ LLM Tracker已安装，跳过重复安装")
            return

        try:
            # 获取模块和函数
            module = __import__(self._module_path, fromlist=[self._func_name])
            original_func = getattr(module, self._func_name)

            # 保存原始函数
            self._original_func = original_func

            # 创建wrapper
            wrapped_func = self._create_wrapper(original_func)

            # 替换
            setattr(module, self._func_name, wrapped_func)

            self._is_installed = True
            print(f"✓ LLM Tracker已安装")
            print(f"  监控函数: {self._module_path}.{self._func_name}")
            print(f"  Token计数: {'✓' if self.count_tokens else '✗'}")
            print(f"  字符计数: {'✓' if self.count_chars else '✗'}")
            print(f"  时间跟踪: {'✓' if self.track_timing else '✗'}")

        except Exception as e:
            print(f"✗ LLM Tracker安装失败: {e}")
            raise

    def uninstall(self) -> None:
        """卸载tracker，恢复原函数"""
        if not self._is_installed:
            print("⚠ LLM Tracker未安装，无需卸载")
            return

        try:
            # 恢复原函数
            module = __import__(self._module_path, fromlist=[self._func_name])
            setattr(module, self._func_name, self._original_func)

            self._is_installed = False
            self._original_func = None
            print("✓ LLM Tracker已卸载，原函数已恢复")

        except Exception as e:
            print(f"✗ LLM Tracker卸载失败: {e}")
            raise

    def _create_wrapper(self, original_func: Callable) -> Callable:
        """创建包装函数"""
        @functools.wraps(original_func)
        async def wrapper(*args, **kwargs):
            # 提取调用信息
            start_time = time.time()

            # 从参数中提取关键信息
            model = kwargs.get('model', args[0] if args else 'unknown')
            prompt = kwargs.get('prompt', args[1] if len(args) > 1 else '')

            # 获取调用者信息（文件、函数名、行号）
            frame = inspect.currentframe()
            caller_info = self._get_caller_info(frame)

            try:
                # 调用原函数
                result = await original_func(*args, **kwargs)

                # 计算耗时
                elapsed = self.track_timing and (time.time() - start_time) or 0

                # 提取token数（如果API响应中有）
                tokens_input = 0
                tokens_output = 0
                if self.count_tokens and hasattr(result, 'usage'):
                    tokens_input = getattr(result.usage, 'prompt_tokens', 0)
                    tokens_output = getattr(result.usage, 'completion_tokens', 0)

                # 记录统计
                self._record_usage(
                    model=model,
                    prompt=prompt,
                    response=result,
                    elapsed=elapsed,
                    tokens_input=tokens_input,
                    tokens_output=tokens_output,
                    caller_info=caller_info
                )

                return result

            except Exception as e:
                # 记录错误
                elapsed = self.track_timing and (time.time() - start_time) or 0
                self._record_error(model, str(e), elapsed, caller_info)
                raise

        return wrapper

    def _get_caller_info(self, frame) -> Dict[str, Any]:
        """获取调用者信息"""
        info = {"file": "unknown", "function": "unknown", "line": 0}

        try:
            # 回溯几层找到实际调用者
            for _ in range(5):  # 最多回溯5层
                frame = frame.f_back
                if frame is None:
                    break

                filename = frame.f_code.co_filename
                # 跳过tracker自身的调用
                if "llm_usage_tracker" not in filename and "llm_tracker" not in filename:
                    info["file"] = os.path.basename(filename)
                    info["function"] = frame.f_code.co_name
                    info["line"] = frame.f_lineno
                    break
        finally:
            del frame

        return info

    def _record_usage(self, model: str, prompt: str, response: Any,
                     elapsed: float, tokens_input: int, tokens_output: int,
                     caller_info: Dict[str, Any]) -> None:
        """记录使用统计"""
        # 基本统计
        self.stats["total_calls"] += 1

        if self.count_chars:
            chars_input = len(prompt)
            chars_output = len(str(response))
            self.stats["total_chars_input"] += chars_input
            self.stats["total_chars_output"] += chars_output

        if self.count_tokens:
            self.stats["total_tokens_input"] += tokens_input
            self.stats["total_tokens_output"] += tokens_output

        if self.track_timing:
            self.stats["total_time"] += elapsed

        # 按模型统计
        if model not in self.stats["calls_by_model"]:
            self.stats["calls_by_model"][model] = {
                "count": 0, "tokens_input": 0, "tokens_output": 0,
                "chars_input": 0, "chars_output": 0, "time": 0.0
            }

        model_stats = self.stats["calls_by_model"][model]
        model_stats["count"] += 1
        model_stats["tokens_input"] += tokens_input
        model_stats["tokens_output"] += tokens_output
        model_stats["chars_input"] += self.count_chars and len(prompt) or 0
        model_stats["chars_output"] += self.count_chars and len(str(response)) or 0
        model_stats["time"] += elapsed

        # 按文件统计
        file_key = caller_info["file"]
        if file_key not in self.stats["calls_by_file"]:
            self.stats["calls_by_file"][file_key] = {"count": 0, "tokens": 0}
        self.stats["calls_by_file"][file_key]["count"] += 1
        self.stats["calls_by_file"][file_key]["tokens"] += tokens_input + tokens_output

        # 按函数统计
        func_key = f"{caller_info['function']} ({caller_info['file']}:{caller_info['line']})"
        if func_key not in self.stats["calls_by_function"]:
            self.stats["calls_by_function"][func_key] = {"count": 0, "tokens": 0}
        self.stats["calls_by_function"][func_key]["count"] += 1
        self.stats["calls_by_function"][func_key]["tokens"] += tokens_input + tokens_output

        # 时间线记录
        self.stats["timeline"].append({
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "elapsed": elapsed,
            "caller": file_key
        })

        # 导出到文件（可选）
        if self.export_to_file:
            self._export_stats()

    def _record_error(self, model: str, error: str, elapsed: float, caller_info: Dict) -> None:
        """记录错误"""
        self.stats["errors"].append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model,
            "error": error,
            "elapsed": elapsed,
            "caller": f"{caller_info['file']}:{caller_info['line']}"
        })

    def _export_stats(self) -> None:
        """导出统计到文件"""
        try:
            with open(self.export_path, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠ 导出统计失败: {e}")

    def get_report(self, detailed: bool = True) -> str:
        """生成使用报告"""
        report = []
        report.append("=" * 70)
        report.append("LLM API 使用统计报告")
        report.append("=" * 70)
        report.append(f"会话开始时间: {self.stats['session_start']}")
        report.append(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # 总体统计
        report.append("【总体统计】")
        report.append(f"  总调用次数: {self.stats['total_calls']:,}")

        if self.count_tokens:
            total_tokens = self.stats['total_tokens_input'] + self.stats['total_tokens_output']
            report.append(f"  总Token数: {total_tokens:,} (输入: {self.stats['total_tokens_input']:,}, 输出: {self.stats['total_tokens_output']:,})")

        if self.count_chars:
            total_chars = self.stats['total_chars_input'] + self.stats['total_chars_output']
            report.append(f"  总字符数: {total_chars:,} (输入: {self.stats['total_chars_input']:,}, 输出: {self.stats['total_chars_output']:,})")

        if self.track_timing:
            avg_time = self.stats['total_time'] / max(1, self.stats['total_calls'])
            report.append(f"  总耗时: {self.stats['total_time']:.2f}秒")
            report.append(f"  平均响应时间: {avg_time:.3f}秒")

        report.append("")

        # 按模型统计
        if self.stats["calls_by_model"]:
            report.append("【按模型统计】")
            for model, stats in self.stats["calls_by_model"].items():
                report.append(f"  📊 {model}:")
                report.append(f"     调用次数: {stats['count']}")

                if self.count_tokens:
                    tokens = stats['tokens_input'] + stats['tokens_output']
                    report.append(f"     Token数: {tokens:,} (输入: {stats['tokens_input']:,}, 输出: {stats['tokens_output']:,})")

                if self.count_chars:
                    chars = stats['chars_input'] + stats['chars_output']
                    report.append(f"     字符数: {chars:,} (输入: {stats['chars_input']:,}, 输出: {stats['chars_output']:,})")

                if self.track_timing:
                    report.append(f"     耗时: {stats['time']:.2f}秒 ({stats['time']/max(1, stats['count']):.3f}秒/次)")

                report.append("")

        # 按文件统计
        if self.stats["calls_by_file"] and detailed:
            report.append("【按文件统计】")
            for file, stats in sorted(self.stats["calls_by_file"].items(),
                                    key=lambda x: x[1]['count'], reverse=True):
                report.append(f"  📁 {file}: {stats['count']} 次调用, {stats['tokens']:,} tokens")
            report.append("")

        # 最近调用时间线
        if self.stats["timeline"] and detailed:
            report.append("【最近调用】")
            for entry in self.stats["timeline"][-10:]:  # 最近10次
                report.append(f"  {entry['timestamp']} - {entry['model']} ({entry['caller']})")
                report.append(f"    Tokens: {entry['tokens_input']:,} → {entry['tokens_output']:,}, "
                            f"耗时: {entry['elapsed']:.2f}s")
            report.append("")

        # 错误统计
        if self.stats["errors"]:
            report.append("【错误统计】")
            report.append(f"  错误次数: {len(self.stats['errors'])}")
            for error in self.stats["errors"][-5:]:  # 最近5个错误
                report.append(f"  {error['timestamp']} - {error['model']} ({error['caller']})")
                report.append(f"    {error['error'][:100]}...")
            report.append("")

        return "\n".join(report)

    def reset(self) -> None:
        """重置统计"""
        self.stats = {
            "session_start": datetime.now().isoformat(),
            "total_calls": 0,
            "total_tokens_input": 0,
            "total_tokens_output": 0,
            "total_chars_input": 0,
            "total_chars_output": 0,
            "total_time": 0.0,
            "calls_by_model": {},
            "calls_by_file": {},
            "calls_by_function": {},
            "errors": [],
            "timeline": []
        }
        print("✓ 统计已重置")

    def is_installed(self) -> bool:
        """检查是否已安装"""
        return self._is_installed


# 创建全局实例
_tracker = None

def get_tracker():
    """获取或创建tracker实例"""
    global _tracker
    if _tracker is None:
        _tracker = LLMUsageTracker(
            count_tokens=True,
            count_chars=True,
            track_timing=True,
            export_to_file=True,  # 自动导出到文件
            export_path="./llm_usage_stats.json"
        )
    return _tracker

def init_tracker():
    """根据环境变量初始化tracker"""
    # 检查是否启用（默认关闭，避免影响生产）
    if os.getenv("ENABLE_LLM_TRACKING", "false").lower() == "true":
        tracker = get_tracker()
        # ⚠️ 重要：必须在导入目标函数之前安装Hook，否则已导入的函数引用不会被更新
        tracker.install()
        print("✓ LLM Tracker 已启用 (ENABLE_LLM_TRACKING=true)")
        print(f"  监控函数: lightrag.llm.openai.openai_complete_if_cache")
        return True
    else:
        print("ℹ LLM Tracker 未启用 (设置 ENABLE_LLM_TRACKING=true 来启用)")
        return False

def print_report(detailed=True):
    """打印使用报告"""
    tracker = get_tracker()
    if tracker.is_installed():
        print(tracker.get_report(detailed))
    else:
        print("⚠ LLM Tracker 未安装")

def reset_stats():
    """重置统计"""
    tracker = get_tracker()
    tracker.reset()

def cleanup():
    """清理（导出文件、卸载等）"""
    tracker = get_tracker()
    if tracker.is_installed():
        print("\n" + "="*70)
        print_report()
        print("="*70 + "\n")
        tracker.uninstall()