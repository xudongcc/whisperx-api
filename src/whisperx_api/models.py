from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, ValidationError, ValidationInfo

from .config import HF_TOKEN


class TranscriptionFormParams(BaseModel):
    """转录请求的表单参数验证模型"""
    model: str = Field(
        default="large-v3-turbo",
        description="要使用的模型名称（兼容性参数，后端始终使用 large-v3-turbo）"
    )
    language: Optional[str] = Field(
        default=None,
        description="音频语言代码（如 'zh', 'en'），如果为空则自动检测",
        max_length=5
    )
    prompt: Optional[str] = Field(
        default=None,
        description="可选的提示文本",
        max_length=1000
    )
    response_format: str = Field(
        default="json",
        description="响应格式",
        pattern="^(json|verbose_json)$"
    )
    temperature: float = Field(
        default=0.0,
        description="采样温度",
        ge=0.0,
        le=1.0
    )
    timestamp_granularities: Optional[str] = Field(
        default=None,
        description="时间戳粒度（暂未使用）"
    )
    enable_speaker_diarization: bool = Field(
        default=True,
        description="是否启用说话人识别"
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="最少说话人数量",
        ge=1,
        le=20
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="最多说话人数量",
        ge=1,
        le=20
    )
    chunk_size: int = Field(
        default=6,
        description="音频分块大小（秒）",
        ge=1,
        le=64
    )
    
    @field_validator('min_speakers', 'max_speakers')
    def validate_speaker_counts(cls, v, info: ValidationInfo):
        """验证说话人数量参数"""
        if v is not None and not info.data.get('enable_speaker_diarization', True):
            from pydantic_core import PydanticCustomError
            raise PydanticCustomError(
                'speaker_count_invalid',
                '说话人数量参数只能在启用说话人识别时使用',
            )
        return v
    
    @field_validator('max_speakers')
    def validate_max_speakers_greater_than_min(cls, v, info: ValidationInfo):
        """验证最大说话人数量应大于等于最小说话人数量"""
        min_speakers = info.data.get('min_speakers')
        if v is not None and min_speakers is not None and v < min_speakers:
            from pydantic_core import PydanticCustomError
            raise PydanticCustomError(
                'max_speakers_invalid',
                '最大说话人数量必须大于等于最小说话人数量',
            )
        return v
    
    @field_validator('enable_speaker_diarization')
    def validate_speaker_diarization_availability(cls, v):
        """验证说话人识别是否可用"""
        if v and not HF_TOKEN:
            from pydantic_core import PydanticCustomError
            raise PydanticCustomError(
                'hf_token_required',
                '说话人识别功能需要设置 HF_TOKEN 环境变量',
            )
        return v


class TranscriptionRequest(BaseModel):
    """完整的转录请求模型（用于文档和类型提示）"""
    file: str = Field(description="音频文件")
    model: str = Field(
        default="large-v3-turbo",
        description="要使用的模型名称"
    )
    language: Optional[str] = Field(
        default=None,
        description="音频语言代码"
    )
    prompt: Optional[str] = Field(
        default=None,
        description="可选的提示文本"
    )
    response_format: str = Field(
        default="json",
        description="响应格式"
    )
    temperature: float = Field(
        default=0.0,
        description="采样温度"
    )
    timestamp_granularities: Optional[List[str]] = Field(
        default=None,
        description="时间戳粒度"
    )
    enable_speaker_diarization: bool = Field(
        default=False,
        description="是否启用说话人识别"
    )
    min_speakers: Optional[int] = Field(
        default=None,
        description="最少说话人数量"
    )
    max_speakers: Optional[int] = Field(
        default=None,
        description="最多说话人数量"
    )
    chunk_size: int = Field(
        default=6,
        description="音频分块大小（秒）"
    )


class SegmentModel(BaseModel):
    """音频片段模型"""
    id: int = Field(description="片段ID")
    start: float = Field(description="开始时间（秒）", ge=0)
    end: float = Field(description="结束时间（秒）", ge=0)
    text: str = Field(description="转录文本")
    avg_logprob: Optional[float] = Field(description="平均对数概率")
    compression_ratio: Optional[float] = Field(description="压缩比")
    no_speech_prob: Optional[float] = Field(description="无语音概率")
    speaker: Optional[str] = Field(description="说话人标识", default=None)
    
    @field_validator('end')
    def validate_end_greater_than_start(cls, v, info: ValidationInfo):
        """验证结束时间大于开始时间"""
        start = info.data.get('start')
        if start is not None and v <= start:
            from pydantic_core import PydanticCustomError
            raise PydanticCustomError(
                'end_time_invalid',
                '结束时间必须大于开始时间',
            )
        return v


class TranscriptionResponse(BaseModel):
    """转录响应模型"""
    text: str = Field(description="完整转录文本")
    language: str = Field(description="检测到的语言")
    duration: float = Field(description="音频总时长（秒）", ge=0)
    segments: List[SegmentModel] = Field(description="音频片段列表")


class VerboseTranscriptionResponse(TranscriptionResponse):
    """详细转录响应模型"""
    task: str = Field(default="transcribe", description="任务类型") 
