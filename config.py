#!/usr/bin/env python3
"""
WhisperX API 配置模块
包含所有共享的配置、日志设置和工具函数
"""

import os
import torch
import logging
from dotenv import load_dotenv
from pythonjsonlogger import jsonlogger

# 加载环境变量
load_dotenv()

# 环境变量配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_NAME = "large-v3-turbo"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "100"))  # MB

def setup_json_logging():
    """设置 JSON 日志配置"""
    # 创建根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, LOG_LEVEL.upper()))
    
    # 设置 JSON 格式化器
    # 定义日志格式字段
    log_format = '%(timestamp)s %(levelname)s %(name)s %(message)s %(module)s %(funcName)s'
    formatter = jsonlogger.JsonFormatter(
        log_format,
        timestamp=True,
        json_ensure_ascii=False
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到根日志记录器
    root_logger.addHandler(console_handler)
    
    # 配置第三方库的日志级别和格式
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(getattr(logging, LOG_LEVEL.upper()))
        # 清除现有处理器
        for handler in logger_instance.handlers[:]:
            logger_instance.removeHandler(handler)
        # 添加JSON格式处理器
        json_handler = logging.StreamHandler()
        json_handler.setFormatter(formatter)
        logger_instance.addHandler(json_handler)
        logger_instance.propagate = False  # 防止重复日志

def get_device():
    """获取计算设备"""
    return "cuda" if torch.cuda.is_available() else "cpu"

def is_speaker_diarization_available():
    """检查说话人识别是否可用"""
    return HF_TOKEN is not None

def get_compute_type():
    """获取计算类型，根据设备自动选择"""
    device = get_device()
    return "float16" if device == "cuda" else "float32"

# 全局设备变量
device = get_device() 
