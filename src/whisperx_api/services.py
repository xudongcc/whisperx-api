import time
import logging
import whisperx

from .config import HF_TOKEN, device, get_compute_type

logger = logging.getLogger(__name__)

# 全局变量存储模型
global_whisper_model = None
global_diarize_model = None


def load_whisper_model():
    """加载 WhisperX 模型"""
    global global_whisper_model
    if global_whisper_model is None:
        start_time = time.time()
        logger.info(
            "Starting WhisperX model loading",
            extra={
                "event": "model_loading_start",
                "model_type": "whisper",
                "model_name": "large-v3-turbo",
                "device": device,
                "compute_type": get_compute_type()
            }
        )
        
        global_whisper_model = whisperx.load_model("large-v3-turbo", device, compute_type=get_compute_type())
        load_time = time.time() - start_time
        
        logger.info(
            "WhisperX model loaded successfully",
            extra={
                "event": "model_loading_complete",
                "model_type": "whisper",
                "model_name": "large-v3-turbo",
                "device": device,
                "compute_type": get_compute_type(),
                "load_time_seconds": round(load_time, 2)
            }
        )
    return global_whisper_model


def load_diarize_model():
    """加载说话人识别模型"""
    global global_diarize_model
    if global_diarize_model is None and HF_TOKEN:
        try:
            start_time = time.time()
            logger.info(
                "Starting speaker diarization model loading",
                extra={
                    "event": "model_loading_start",
                    "model_type": "diarization",
                    "device": device,
                    "has_hf_token": bool(HF_TOKEN)
                }
            )
            
            global_diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
            load_time = time.time() - start_time
            
            logger.info(
                "Speaker diarization model loaded successfully",
                extra={
                    "event": "model_loading_complete",
                    "model_type": "diarization",
                    "device": device,
                    "load_time_seconds": round(load_time, 2)
                }
            )
        except Exception as e:
            logger.error(
                "Failed to load diarization model",
                extra={
                    "event": "model_loading_error",
                    "model_type": "diarization",
                    "device": device,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            global_diarize_model = None
    return global_diarize_model


def get_whisper_model():
    """获取 WhisperX 模型实例"""
    return global_whisper_model


def get_diarize_model():
    """获取说话人识别模型实例"""
    return global_diarize_model


def is_whisper_model_loaded():
    """检查 WhisperX 模型是否已加载"""
    return global_whisper_model is not None


def is_diarize_model_loaded():
    """检查说话人识别模型是否已加载"""
    return global_diarize_model is not None 
