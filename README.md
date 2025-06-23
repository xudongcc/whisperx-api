# WhisperX API

一个兼容 OpenAI Whisper API 的 WhisperX 服务，支持说话人识别功能。

## 功能特性

- 🎯 兼容 OpenAI Whisper API 格式
- 🎤 支持说话人识别（Speaker Diarization）
- ⚡ 高性能音频转录
- 🔄 时间戳对齐
- 🌍 多语言支持

## 安装

1. 克隆仓库：

```bash
git clone <repository-url>
cd whisperx-api
```

2. 使用 uv 安装依赖：

```bash
uv sync
```

3. 配置环境变量：

```bash
# 复制环境变量模板
cp env.example .env

# 编辑 .env 文件，设置你的配置
# 特别是 HF_TOKEN（用于说话人识别功能）
```

### 验证配置

```bash
# 测试配置是否正确加载
uv run python test_config.py
```

## 使用方法

### 启动服务

```bash
# 使用 uv 运行
uv run python app.py

# 或者直接运行
python app.py
```

服务将在 `http://localhost:8000` 启动（可通过环境变量 HOST 和 PORT 配置）。

### API 端点

#### 1. 基础转录（无说话人识别）

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "model=large-v3-turbo" \
  -F "response_format=json"
```

#### 2. 带说话人识别的转录

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "model=large-v3-turbo" \
  -F "response_format=json" \
  -F "enable_speaker_diarization=true" \
  -F "min_speakers=2" \
  -F "max_speakers=4"
```

#### 3. 详细响应格式

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3" \
  -F "model=large-v3-turbo" \
  -F "response_format=verbose_json" \
  -F "enable_speaker_diarization=true"
```

### 参数说明

| 参数                         | 类型    | 必需 | 默认值           | 说明                                  |
| ---------------------------- | ------- | ---- | ---------------- | ------------------------------------- |
| `file`                       | File    | ✅   | -                | 音频文件                              |
| `model`                      | string  | ❌   | "large-v3-turbo" | 模型名称（目前只支持 large-v3-turbo） |
| `language`                   | string  | ❌   | None             | 语言代码（如 "zh", "en"）             |
| `response_format`            | string  | ❌   | "json"           | 响应格式（"json" 或 "verbose_json"）  |
| `enable_speaker_diarization` | boolean | ❌   | false            | 是否启用说话人识别                    |
| `min_speakers`               | integer | ❌   | None             | 最小说话人数量                        |
| `max_speakers`               | integer | ❌   | None             | 最大说话人数量                        |

### 响应格式

#### 标准 JSON 响应

```json
{
  "text": "完整的转录文本",
  "language": "zh",
  "duration": 120.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "第一段文本",
      "speaker": "SPEAKER_00",
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.1
    }
  ]
}
```

#### 详细 JSON 响应

```json
{
  "task": "transcribe",
  "language": "zh",
  "duration": 120.5,
  "text": "完整的转录文本",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 5.2,
      "text": "第一段文本",
      "tokens": [1, 2, 3, 4],
      "temperature": 0.0,
      "speaker": "SPEAKER_00",
      "avg_logprob": -0.5,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.1
    }
  ]
}
```

## 健康检查

```bash
curl http://localhost:8000/health
```

响应：

```json
{
  "status": "healthy",
  "model_loaded": true,
  "diarization_model_loaded": true,
  "hf_token_available": true,
  "device": "cuda",
  "compute_type": "float16",
  "host": "0.0.0.0",
  "port": 8000,
  "max_file_size_mb": 100,
  "log_level": "INFO",
  "debug": false
}
```

## 环境变量

| 变量名          | 说明                                         | 必需                 | 默认值  |
| --------------- | -------------------------------------------- | -------------------- | ------- |
| `HF_TOKEN`      | HuggingFace 访问令牌，用于下载说话人识别模型 | 启用说话人识别时必需 | -       |
| `HOST`          | 服务器监听地址                               | ❌                   | 0.0.0.0 |
| `PORT`          | 服务器监听端口                               | ❌                   | 8000    |
| `MAX_FILE_SIZE` | 最大文件大小（MB）                           | ❌                   | 100     |
| `LOG_LEVEL`     | 日志级别                                     | ❌                   | INFO    |
| `DEBUG`         | 调试模式                                     | ❌                   | false   |

### 环境变量配置示例

创建 `.env` 文件：

```bash
# 复制模板文件
cp env.example .env

# 编辑 .env 文件
nano .env
```

`.env` 文件内容示例：

```env
# HuggingFace 访问令牌
HF_TOKEN=hf_your_token_here

# 服务器配置
HOST=0.0.0.0
PORT=8000

# 计算类型现在根据设备自动选择 (CUDA: float16, CPU: float32)

# 文件上传配置
MAX_FILE_SIZE=100

# 日志配置
LOG_LEVEL=INFO

# 开发环境配置
DEBUG=false
```

## 注意事项

1. **说话人识别**：需要设置 `HF_TOKEN` 环境变量才能使用说话人识别功能
2. **模型大小**：large-v3-turbo 模型较大，首次加载可能需要一些时间
3. **CUDA 支持**：建议使用 CUDA 以获得更好的性能
4. **音频格式**：支持常见的音频格式（mp3, wav, m4a 等）

## 错误处理

- 如果说话人识别失败，API 会继续返回转录结果，但不包含说话人标签
- 所有错误都会在日志中记录详细信息
- HTTP 状态码会正确反映错误类型

## 开发

### 本地开发

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

# 代码格式化
uv run black .
uv run isort .

# 代码检查
uv run flake8 .

# 启动开发服务器
uv run uvicorn app:app --reload
```

### Docker 部署

```bash
# 构建镜像
docker build -t whisperx-api .

# 运行容器
docker run -p 8000:8000 --env-file .env whisperx-api
```
