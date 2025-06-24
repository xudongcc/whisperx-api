import os
import tempfile
import uuid
from pathlib import Path
from typing import Union
from contextlib import asynccontextmanager
import time
import logging

import whisperx
from fastapi import FastAPI, File, HTTPException, UploadFile, Request, Depends
from fastapi.responses import JSONResponse

# 导入共享配置
from .config import (
    HF_TOKEN, device, get_compute_type
)
from .models import (
    TranscriptionFormParams, TranscriptionResponse, VerboseTranscriptionResponse
)
from .services import load_whisper_model, load_diarize_model, is_whisper_model_loaded, is_diarize_model_loaded
from .validators import validate_transcription_params, validate_audio_file, validate_file_size

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    load_whisper_model()
    # 预加载说话人识别模型（如果HF_TOKEN可用）
    if HF_TOKEN:
        load_diarize_model()
    yield


app = FastAPI(
    title="WhisperX API",
    description="WhisperX API compatible with OpenAI Whisper API with speaker diarization",
    version="1.0.0",
    lifespan=lifespan
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录HTTP访问日志的中间件"""
    start_time = time.time()
    
    # 获取客户端信息
    client_ip = request.client.host if request.client else "unknown"
    client_port = request.client.port if request.client else 0
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # 记录访问日志
    logger.info(
        "HTTP request processed",
        extra={
            "event": "http_access",
            "client_addr": f"{client_ip}:{client_port}",
            "method": request.method,
            "path": str(request.url.path),
            "query_params": str(request.url.query) if request.url.query else None,
            "status_code": response.status_code,
            "process_time_seconds": round(process_time, 4),
            "user_agent": request.headers.get("user-agent"),
            "content_type": request.headers.get("content-type"),
            "content_length": request.headers.get("content-length")
        }
    )
    
    return response


@app.get("/")
async def root():
    """根路径"""
    return {"message": "WhisperX API is running", "model": "large-v3-turbo", "device": device}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy", 
        "model_loaded": is_whisper_model_loaded(),
        "diarization_model_loaded": is_diarize_model_loaded(),
        "hf_token_available": HF_TOKEN is not None,
        "device": device,
        "compute_type": get_compute_type()
    }


@app.post("/v1/audio/transcriptions", response_model=Union[TranscriptionResponse, VerboseTranscriptionResponse])
async def create_transcription(
    file: UploadFile = File(..., description="要转录的音频或视频文件"),
    params: TranscriptionFormParams = Depends(validate_transcription_params)
):
    """
    创建音频转录 - 兼容 OpenAI Whisper API，支持说话人识别
    
    支持的文件格式:
    - 音频: mp3, wav, flac, ogg, m4a, aac
    - 视频: mp4, mov, avi, mkv, webm
    
    参数说明:
    - file: 音频或视频文件
    - model: 模型名称（兼容性参数，可传入任意值，后端始终使用 large-v3-turbo）
    - language: 语言代码（可选，自动检测）
    - response_format: 响应格式（json 或 verbose_json）
    - temperature: 采样温度（0.0-1.0）
    - enable_speaker_diarization: 是否启用说话人识别
    - min_speakers/max_speakers: 说话人数量范围
    - chunk_size: 音频分块大小（1-64秒）
    """
    # 生成请求ID用于跟踪
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    try:
        # 使用新的验证函数验证音频文件
        validate_audio_file(file)
        
        # 验证文件大小并获取内容
        content = await validate_file_size(file)
        file_size_mb = len(content) / (1024 * 1024)
        
        logger.info(
            "Transcription request received with validated parameters",
            extra={
                "event": "transcription_request_start",
                "request_id": request_id,
                "file_name": file.filename,
                "content_type": file.content_type,
                "file_size_mb": round(file_size_mb, 2),
                "requested_model": params.model,
                "actual_model": "large-v3-turbo",
                "validated_params": params.model_dump()
            }
        )
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 加载模型
            whisper_model = load_whisper_model()
            
            # 转录音频
            transcribe_start_time = time.time()
            logger.info(
                "Starting audio transcription with validated parameters",
                extra={
                    "event": "transcription_start",
                    "request_id": request_id,
                    "file_name": file.filename,
                    "temp_file_path": temp_file_path,
                    "language": params.language,
                    "chunk_size": params.chunk_size
                }
            )
            
            transcribe_result = whisper_model.transcribe(
                temp_file_path,
                language=params.language,
                chunk_size=params.chunk_size,
            )
            transcribe_time = time.time() - transcribe_start_time
            
            logger.info(
                "Audio transcription completed",
                extra={
                    "event": "transcription_complete",
                    "request_id": request_id,
                    "file_name": file.filename,
                    "detected_language": transcribe_result.get('language'),
                    "segment_count": len(transcribe_result.get('segments', [])),
                    "transcribe_time_seconds": round(transcribe_time, 2)
                }
            )
            
            # 对齐时间戳
            align_start_time = time.time()
            logger.info(
                "Starting timestamp alignment",
                extra={
                    "event": "alignment_start",
                    "request_id": request_id,
                    "language": transcribe_result['language'],
                    "device": device,
                    "segment_count": len(transcribe_result['segments'])
                }
            )
            
            try:
                align_model, metadata = whisperx.load_align_model(language_code=transcribe_result["language"], device=device)
                logger.info(
                    "Alignment model loaded",
                    extra={
                        "event": "alignment_model_loaded",
                        "request_id": request_id,
                        "language": transcribe_result["language"],
                        "device": device
                    }
                )
                
                align_result = whisperx.align(transcribe_result["segments"], align_model, metadata, temp_file_path, device)
                align_time = time.time() - align_start_time
                
                logger.info(
                    "Timestamp alignment completed",
                    extra={
                        "event": "alignment_complete",
                        "request_id": request_id,
                        "language": transcribe_result["language"],
                        "segment_count": len(align_result.get("segments", [])),
                        "align_time_seconds": round(align_time, 2)
                    }
                )
                
            except Exception as align_error:
                align_time = time.time() - align_start_time
                logger.error(
                    "Timestamp alignment failed",
                    extra={
                        "event": "alignment_error",
                        "request_id": request_id,
                        "language_code": transcribe_result.get('language'),
                        "device": device,
                        "segment_count": len(transcribe_result.get('segments', [])),
                        "error_type": type(align_error).__name__,
                        "error_message": str(align_error),
                        "align_time_seconds": round(align_time, 2)
                    }
                )
                
                # 如果对齐失败，继续使用原始结果，但记录警告
                logger.warning(
                    "Continuing with original transcription without alignment",
                    extra={
                        "event": "alignment_fallback",
                        "request_id": request_id,
                        "language": transcribe_result.get('language'),
                        "segment_count": len(transcribe_result.get('segments', []))
                    }
                )
                
                # 确保result仍然包含必要的字段
                if "segments" not in transcribe_result:
                    transcribe_result["segments"] = []
                align_result = transcribe_result
            
            # 说话人识别
            if params.enable_speaker_diarization:
                diarize_start_time = time.time()
                try:
                    # 加载说话人识别模型
                    diarize_model = load_diarize_model()
                    
                    logger.info(
                        "Starting speaker diarization",
                        extra={
                            "event": "diarization_start",
                            "request_id": request_id,
                            "min_speakers": params.min_speakers,
                            "max_speakers": params.max_speakers,
                            "has_diarize_model": diarize_model is not None
                        }
                    )
                    
                    if diarize_model:
                        # 执行说话人识别
                        if params.min_speakers is not None and params.max_speakers is not None:
                            diarize_segments = diarize_model(temp_file_path, min_speakers=params.min_speakers, max_speakers=params.max_speakers)
                        else:
                            diarize_segments = diarize_model(temp_file_path)
                        
                        logger.info(
                            "Speaker diarization analysis completed",
                            extra={
                                "event": "diarization_analysis_complete",
                                "request_id": request_id,
                                "speaker_segment_count": len(diarize_segments),
                                "min_speakers": params.min_speakers,
                                "max_speakers": params.max_speakers
                            }
                        )
                        
                        # 分配说话人标签
                        logger.info(
                            "Assigning speaker labels to transcription segments",
                            extra={
                                "event": "speaker_assignment_start",
                                "request_id": request_id,
                                "transcription_segment_count": len(align_result.get("segments", [])),
                                "speaker_segment_count": len(diarize_segments)
                            }
                        )
                        
                        assign_word_speakers_result = whisperx.assign_word_speakers(diarize_segments, align_result)
                        diarize_time = time.time() - diarize_start_time
                        
                        # 统计说话人信息
                        speakers = set()
                        for segment in assign_word_speakers_result["segments"]:
                            if "speaker" in segment:
                                speakers.add(segment["speaker"])
                        
                        logger.info(
                            "Speaker diarization completed successfully",
                            extra={
                                "event": "diarization_complete",
                                "request_id": request_id,
                                "identified_speakers": sorted(list(speakers)),
                                "speaker_count": len(speakers),
                                "final_segment_count": len(assign_word_speakers_result.get("segments", [])),
                                "diarize_time_seconds": round(diarize_time, 2)
                            }
                        )
                        
                        # 使用包含说话人信息的结果
                        final_result = assign_word_speakers_result
                    else:
                        diarize_time = time.time() - diarize_start_time
                        logger.warning(
                            "Diarization model not available",
                            extra={
                                "event": "diarization_model_unavailable",
                                "request_id": request_id,
                                "has_hf_token": bool(HF_TOKEN),
                                "diarize_time_seconds": round(diarize_time, 2)
                            }
                        )
                        final_result = align_result
                        
                except Exception as diarize_error:
                    diarize_time = time.time() - diarize_start_time
                    logger.error(
                        "Speaker diarization failed",
                        extra={
                            "event": "diarization_error",
                            "request_id": request_id,
                            "error_type": type(diarize_error).__name__,
                            "error_message": str(diarize_error),
                            "min_speakers": params.min_speakers,
                            "max_speakers": params.max_speakers,
                            "diarize_time_seconds": round(diarize_time, 2)
                        }
                    )
                    logger.warning(
                        "Continuing without speaker labels",
                        extra={
                            "event": "diarization_fallback",
                            "request_id": request_id
                        }
                    )
                    final_result = align_result
            else:
                final_result = align_result
            
            # 计算总时长
            duration = max(segment["end"] for segment in final_result["segments"]) if final_result["segments"] else 0.0
            
            # 合并所有文本
            full_text = " ".join(segment["text"] for segment in final_result["segments"])
            
            if params.response_format == "verbose_json":
                # 格式化详细响应
                segments = []
                for segment in final_result["segments"]:
                    segment_data = {
                        "id": len(segments),
                        "seek": 0,
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "tokens": segment.get("tokens", []),
                        "temperature": params.temperature,
                        "avg_logprob": segment.get("avg_logprob", 0.0),
                        "compression_ratio": segment.get("compression_ratio", 0.0),
                        "no_speech_prob": segment.get("no_speech_prob", 0.0)
                    }
                    
                    # 添加说话人标签（如果可用）
                    if params.enable_speaker_diarization and "speaker" in segment:
                        segment_data["speaker"] = segment["speaker"]
                    
                    segments.append(segment_data)
                
                response_data = {
                    "task": "transcribe",
                    "language": transcribe_result["language"],
                    "duration": duration,
                    "text": full_text,
                    "segments": segments
                }
                
                total_time = time.time() - start_time
                logger.info(
                    "Verbose transcription request completed",
                    extra={
                        "event": "transcription_request_complete",
                        "request_id": request_id,
                        "file_name": file.filename,
                        "response_format": params.response_format,
                        "language": transcribe_result["language"],
                        "duration_seconds": round(duration, 2),
                        "segment_count": len(segments),
                        "text_length": len(full_text),
                        "speaker_diarization_enabled": params.enable_speaker_diarization,
                        "identified_speakers": len(set(s.get("speaker") for s in segments if s.get("speaker"))),
                        "total_processing_time_seconds": round(total_time, 2)
                    }
                )
            else:
                # 格式化标准响应
                segments = []
                for segment in final_result["segments"]:
                    segment_data = {
                        "id": len(segments),
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"].strip(),
                        "avg_logprob": segment.get("avg_logprob", 0.0),
                        "compression_ratio": segment.get("compression_ratio", 0.0),
                        "no_speech_prob": segment.get("no_speech_prob", 0.0)
                    }
                    
                    # 添加说话人标签（如果可用）
                    if params.enable_speaker_diarization and "speaker" in segment:
                        segment_data["speaker"] = segment["speaker"]
                    
                    segments.append(segment_data)
                
                response_data = {
                    "text": full_text,
                    "language": transcribe_result["language"],
                    "duration": duration,
                    "segments": segments
                }
                
                total_time = time.time() - start_time
                logger.info(
                    "Standard transcription request completed",
                    extra={
                        "event": "transcription_request_complete",
                        "request_id": request_id,
                        "file_name": file.filename,
                        "response_format": params.response_format,
                        "language": transcribe_result["language"],
                        "duration_seconds": round(duration, 2),
                        "segment_count": len(segments),
                        "text_length": len(full_text),
                        "speaker_diarization_enabled": params.enable_speaker_diarization,
                        "identified_speakers": len(set(s.get("speaker") for s in segments if s.get("speaker"))),
                        "total_processing_time_seconds": round(total_time, 2)
                    }
                )
            
            return JSONResponse(content=response_data)
            
        finally:
            # 清理临时文件
            try:
                os.unlink(temp_file_path)
                logger.debug(
                    "Temporary file cleaned up",
                    extra={
                        "event": "temp_file_cleanup",
                        "request_id": request_id,
                        "temp_file_path": temp_file_path
                    }
                )
            except Exception as cleanup_error:
                logger.warning(
                    "Failed to cleanup temporary file",
                    extra={
                        "event": "temp_file_cleanup_error",
                        "request_id": request_id,
                        "temp_file_path": temp_file_path,
                        "error_type": type(cleanup_error).__name__,
                        "error_message": str(cleanup_error)
                    }
                )
            
    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        total_time = time.time() - start_time
        logger.error(
            "Transcription request failed with unexpected error",
            extra={
                "event": "transcription_request_error",
                "request_id": request_id,
                "file_name": getattr(file, 'filename', 'unknown'),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "total_processing_time_seconds": round(total_time, 2)
            }
        )
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
