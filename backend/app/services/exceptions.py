"""
SAM3 Demo Backend - Custom Exceptions

This module defines custom exceptions for the services layer.
"""


class SampleNotFoundError(Exception):
    """样本未找到异常
    
    当请求的 sample_id 在缓存中不存在时抛出。
    
    Attributes:
        sample_id: 请求的样本 ID
        message: 错误消息
    """
    
    def __init__(self, sample_id: str, message: str = None):
        self.sample_id = sample_id
        self.message = message or f"Sample not found: {sample_id}"
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return self.message
