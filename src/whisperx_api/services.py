import time
import logging
import whisperx

from .config import HF_TOKEN, device, get_compute_type

logger = logging.getLogger(__name__)

# 全局变量存储模型
global_whisper_model = None
global_diarize_pipeline = None
global_align_models = {}  # 缓存不同语言的对齐模型


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
            "Whisper model loaded successfully",
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


def load_diarize_pipeline():
    """加载说话人识别模型"""
    global global_diarize_pipeline
    if global_diarize_pipeline is None and HF_TOKEN:
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
            
            global_diarize_pipeline = whisperx.diarize.DiarizationPipeline(use_auth_token=HF_TOKEN, device=device)
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
            # 显示具体的错误信息，便于调试
            error_details = f"错误类型: {type(e).__name__}, 错误信息: {str(e)}"
            logger.error(f"说话人识别模型加载失败 - {error_details} (设备: {device})")
            
            # 记录详细的错误信息到结构化日志（如果有JSON处理器）
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
            global_diarize_pipeline = None
    return global_diarize_pipeline


def get_whisper_model():
    """获取 WhisperX 模型实例"""
    return global_whisper_model


def get_diarize_pipeline():
    """获取说话人识别模型实例"""
    return global_diarize_pipeline


def is_whisper_model_loaded():
    """检查 WhisperX 模型是否已加载"""
    return global_whisper_model is not None


def is_diarize_pipeline_loaded():
    """检查说话人识别模型是否已加载"""
    return global_diarize_pipeline is not None


def load_align_model(language_code: str):
    """加载并缓存对齐模型"""
    global global_align_models
    
    if language_code not in global_align_models:
        start_time = time.time()
        logger.info(
            "Loading alignment model for language",
            extra={
                "event": "align_model_loading_start",
                "language": language_code,
                "device": device
            }
        )
        
        try:
            align_model, metadata = whisperx.load_align_model(
                language_code=language_code, 
                device=device
            )
            global_align_models[language_code] = (align_model, metadata)
            load_time = time.time() - start_time
            
            logger.info(
                "Alignment model loaded successfully",
                extra={
                    "event": "align_model_loading_complete",
                    "language": language_code,
                    "device": device,
                    "load_time_seconds": round(load_time, 2),
                    "cached_languages": list(global_align_models.keys())
                }
            )
        except Exception as e:
            # 显示具体的错误信息，便于调试
            error_details = f"错误类型: {type(e).__name__}, 错误信息: {str(e)}"
            logger.error(f"对齐模型加载失败 - {error_details} (语言: {language_code}, 设备: {device})")
            
            # 记录详细的错误信息到结构化日志（如果有JSON处理器） 
            logger.error(
                "Failed to load alignment model",
                extra={
                    "event": "align_model_loading_error",
                    "language": language_code,
                    "device": device,
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
            raise
    
    return global_align_models[language_code]


def get_cached_languages():
    """获取已缓存的语言列表"""
    return list(global_align_models.keys())


def get_gpu_memory_info():
    """获取 GPU 内存使用信息"""
    import torch
    if torch.cuda.is_available():
        # 获取当前 GPU 内存使用情况
        allocated = torch.cuda.memory_allocated() / 1024**3  # 转换为 GB
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
            "free_gb": round(total - allocated, 2),
            "usage_percent": round((allocated / total) * 100, 1)
        }
    return None 
