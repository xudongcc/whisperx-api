from pathlib import Path
from typing import Optional
from pydantic import ValidationError
from fastapi import Form, HTTPException, UploadFile

from .models import TranscriptionFormParams
from .config import MAX_FILE_SIZE


async def validate_transcription_params(
    model: str = Form("large-v3-turbo"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
    enable_speaker_diarization: bool = Form(True),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    chunk_size: int = Form(30)
) -> TranscriptionFormParams:
    """验证转录请求的表单参数"""
    try:
        params = TranscriptionFormParams(
            model=model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            timestamp_granularities=timestamp_granularities,
            enable_speaker_diarization=enable_speaker_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            chunk_size=chunk_size
        )
        return params
    except ValidationError as e:
        # 提取验证错误信息并翻译
        error_messages = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error["loc"])
            message = error["msg"]
            error_messages.append(f"{field}: {message}")
        
        raise HTTPException(
            status_code=422,
            detail={
                "message": "参数验证失败",
                "errors": error_messages,
                "validation_details": e.errors()
            }
        )


def validate_audio_file(file: UploadFile) -> None:
    """验证音频文件"""
    # 验证文件是否存在
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="未提供音频文件")
    
    # 验证文件类型
    if file.content_type:
        if not any(file.content_type.startswith(ct) for ct in ['audio/', 'video/']):
            raise HTTPException(
                status_code=400, 
                detail=f"不支持的文件类型: {file.content_type}。支持的类型: 音频和视频文件"
            )
    
    # 验证文件扩展名
    allowed_extensions = {
        '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', 
        '.mp4', '.mov', '.avi', '.mkv', '.webm'
    }
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件扩展名: {file_extension}。支持的扩展名: {', '.join(sorted(allowed_extensions))}"
        )


async def validate_file_size(file: UploadFile) -> bytes:
    """验证文件大小并返回文件内容"""
    try:
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": f"文件大小超出限制",
                    "file_size_mb": round(file_size_mb, 2),
                    "max_file_size_mb": MAX_FILE_SIZE,
                    "error_code": "FILE_TOO_LARGE"
                }
            )
        
        return content
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"读取文件时发生错误: {str(e)}"
        ) 
