"""
LightRAG 异步计时器模块

提供高性能、低开销的异步函数计时功能，专为分析本地计算瓶颈设计。
支持配置文件开关，最小化对程序性能的影响。
"""

import asyncio
import time
import functools
import os
from typing import Callable, Any, Optional, Dict, List
from pathlib import Path
from dotenv import load_dotenv

# 默认配置
DEFAULT_CONFIG = {
    "enable_timing": False,  # 默认关闭，需要通过.env文件开启
    "min_duration_ms": 0.1,  # 只记录超过0.1ms的操作
    "max_call_stack_depth": 10,  # 防止栈溢出
}

class TimingStats:
    """计时统计信息"""
    def __init__(self):
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.last_call_time = 0.0
        self.call_history: List[float] = []
        
    def add_call(self, duration: float):
        """添加一次调用记录"""
        self.call_count += 1
        self.total_time += duration
        self.last_call_time = duration
        
        if duration < self.min_time:
            self.min_time = duration
        if duration > self.max_time:
            self.max_time = duration
            
        # 只保留最近的记录，避免内存泄漏
        if len(self.call_history) < 100:
            self.call_history.append(duration)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        if self.call_count == 0:
            return {"status": "no_calls"}
            
        avg_time = self.total_time / self.call_count
        
        return {
            "call_count": self.call_count,
            "total_time_ms": self.total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": self.min_time * 1000,
            "max_time_ms": self.max_time * 1000,
            "last_call_ms": self.last_call_time * 1000,
            "throughput_per_sec": 1.0 / avg_time if avg_time > 0 else 0,
        }

class AsyncTimer:
    """异步计时器"""
    
    # 全局统计存储
    _global_stats: Dict[str, TimingStats] = {}
    _config_loaded = False
    _config = DEFAULT_CONFIG.copy()
    
    @classmethod
    def _load_config(cls):
        """加载配置文件，只加载一次"""
        if cls._config_loaded:
            return
            
        # 查找项目根目录的.env文件
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # lightrag/utils/timing.py -> lightrag -> 项目根
        env_file = project_root / ".env"
        
        # 如果没有找到，向上一级查找
        if not env_file.exists():
            env_file = Path(__file__).parent.parent / ".env"
        
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            
        # 读取计时器配置
        cls._config["enable_timing"] = (
            os.getenv("LIGHTRAG_ENABLE_TIMING", "false").lower() == "true"
        )
        cls._config["min_duration_ms"] = float(
            os.getenv("LIGHTRAG_TIMING_MIN_DURATION", "0.1")
        )
        cls._config["max_call_stack_depth"] = int(
            os.getenv("LIGHTRAG_TIMING_MAX_DEPTH", "10")
        )
        
        cls._config_loaded = True
    
    @classmethod
    def is_enabled(cls) -> bool:
        """检查计时器是否启用"""
        cls._load_config()
        return cls._config["enable_timing"]
    
    @classmethod
    def get_min_duration(cls) -> float:
        """获取最小记录阈值（秒）"""
        cls._load_config()
        return cls._config["min_duration_ms"] / 1000.0
    
    @classmethod
    def get_max_depth(cls) -> int:
        """获取最大调用栈深度"""
        cls._load_config()
        return cls._config["max_call_stack_depth"]
    
    @classmethod
    def reset_stats(cls):
        """重置所有统计信息"""
        cls._global_stats.clear()
    
    @classmethod
    def get_stats(cls, function_name: str) -> Dict[str, Any]:
        """获取指定函数的统计信息"""
        stats = cls._global_stats.get(function_name)
        if stats:
            return stats.get_stats()
        return {"status": "not_found"}
    
    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有统计信息"""
        return {name: stats.get_stats() for name, stats in cls._global_stats.items()}
    
    @classmethod
    def print_report(cls):
        """打印计时报告"""
        if not cls.is_enabled():
            print("Timing is disabled. Set LIGHTRAG_ENABLE_TIMING=true in .env to enable.")
            return
            
        stats = cls.get_all_stats()
        if not stats:
            print("No timing data collected.")
            return
        
        print("\n=== LightRAG Timing Report ===")
        print(f"{'Function':<40} {'Calls':<8} {'Total(ms)':<12} {'Avg(ms)':<12} {'Max(ms)':<12} {'Throughput':<12}")
        print("-" * 100)
        
        for func_name, func_stats in sorted(stats.items(), key=lambda x: x[1].get("total_time_ms", 0), reverse=True):
            if func_stats.get("status") == "no_calls":
                continue
                
            print(f"{func_name:<40} "
                  f"{func_stats['call_count']:<8} "
                  f"{func_stats['total_time_ms']:<12.2f} "
                  f"{func_stats['avg_time_ms']:<12.2f} "
                  f"{func_stats['max_time_ms']:<12.2f} "
                  f"{func_stats['throughput_per_sec']:<12.2f}")
        
        print("-" * 100)
        print("Tip: 设置 LIGHTRAG_TIMING_MIN_DURATION 来过滤小操作")
    
    @classmethod
    def measure_sync(cls, function_name: str = None):
        """同步函数计时装饰器"""
        def decorator(func: Callable) -> Callable:
            func_name = function_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # 快速检查：避免在函数调用前进行昂贵的配置检查
                if not cls.is_enabled():
                    return func(*args, **kwargs)
                
                start_time = time.perf_counter()
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    # 只记录超过阈值的操作
                    if duration >= cls.get_min_duration():
                        if func_name not in cls._global_stats:
                            cls._global_stats[func_name] = TimingStats()
                        cls._global_stats[func_name].add_call(duration)
            
            return wrapper
        return decorator
    
    @classmethod
    def measure_async(cls, function_name: str = None):
        """异步函数计时装饰器"""
        def decorator(func: Callable) -> Callable:
            func_name = function_name or f"{func.__module__}.{func.__name__}"
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # 快速检查：避免在函数调用前进行昂贵的配置检查
                if not cls.is_enabled():
                    return await func(*args, **kwargs)
                
                start_time = time.perf_counter()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    end_time = time.perf_counter()
                    duration = end_time - start_time
                    
                    # 只记录超过阈值的操作
                    if duration >= cls.get_min_duration():
                        if func_name not in cls._global_stats:
                            cls._global_stats[func_name] = TimingStats()
                        cls._global_stats[func_name].add_call(duration)
            
            return wrapper
        return decorator
    
    @classmethod
    def measure_block(cls, operation_name: str):
        """同步代码块计时器，使用async with语法"""
        if not cls.is_enabled():
            return _NullAsyncContext()
        
        return _AsyncTimerContext(operation_name)
    
    @classmethod
    def measure_async_block(cls, operation_name: str):
        """异步代码块计时器，使用async with语法"""
        if not cls.is_enabled():
            return _NullAsyncContext()
        
        return _AsyncTimerContext(operation_name)


class _AsyncTimerContext:
    """异步计时器上下文管理器"""
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = 0.0
    
    async def __aenter__(self):
        if AsyncTimer.is_enabled():
            self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if AsyncTimer.is_enabled() and self.start_time > 0:
            end_time = time.perf_counter()
            duration = end_time - self.start_time
            
            if duration >= AsyncTimer.get_min_duration():
                if self.operation_name not in AsyncTimer._global_stats:
                    AsyncTimer._global_stats[self.operation_name] = TimingStats()
                AsyncTimer._global_stats[self.operation_name].add_call(duration)


class _NullAsyncContext:
    """空计时器上下文（当计时器禁用时使用）"""
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# 便捷函数
def timeit_sync(func_name: str = None):
    """同步函数计时装饰器"""
    return AsyncTimer.measure_sync(func_name)

def timeit_async(func_name: str = None):
    """异步函数计时装饰器"""
    return AsyncTimer.measure_async(func_name)

def timeit_block(operation_name: str):
    """同步代码块计时器"""
    return AsyncTimer.measure_block(operation_name)

def timeit_async_block(operation_name: str):
    """异步代码块计时器"""
    return AsyncTimer.measure_async_block(operation_name)

def print_timing_report():
    """打印计时报告"""
    AsyncTimer.print_report()