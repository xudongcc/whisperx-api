"""
WhisperX API - 基于 WhisperX 的语音转录 API 服务

支持多语言语音转录和说话人识别功能。
"""

__version__ = "1.0.0"
__author__ = "whisperx-api"
__description__ = "WhisperX API compatible with OpenAI Whisper API with speaker diarization"

from .config import *
from .models import *
from .services import *
from .validators import * 
