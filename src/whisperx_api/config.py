#!/usr/bin/env python3
"""
WhisperX API 配置模块
包含所有共享的配置、日志设置和工具函数
"""

import os
import torch
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 环境变量配置
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MODEL_NAME = "large-v3-turbo"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
HF_TOKEN = os.getenv("HF_TOKEN")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "1024"))  # MB

# 配置日志系统
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
