# =====================================
# 多阶段构建：模型下载阶段
# =====================================
FROM python:3.11-slim AS model-downloader

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 安装系统依赖（模型下载阶段需要）
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 设置工作目录
WORKDIR /app

# 复制项目配置文件
COPY pyproject.toml uv.lock README.md ./

# 安装 Python 依赖（使用 uv）
RUN uv sync --frozen --no-dev

# 预下载 WhisperX 模型
RUN uv run python -c "import whisperx; model = whisperx.load_model('large-v3-turbo', 'cpu', compute_type='float32')"

# 预下载对齐模型（常用语言）
RUN uv run python -c "import whisperx; whisperx.load_align_model('en', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('zh', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('ja', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('ko', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('es', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('fr', 'cpu')" || true
RUN uv run python -c "import whisperx; whisperx.load_align_model('de', 'cpu')" || true

# =====================================
# 生产环境构建阶段
# =====================================
FROM python:3.11-slim AS production

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    # 音频处理依赖
    ffmpeg \
    libsndfile1 \
    # 构建工具
    build-essential \
    # 网络工具
    curl \
    wget \
    # OpenMP 支持
    libgomp1 \
    # 清理工具
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 创建非root用户
RUN groupadd -r whisperx && useradd -r -g whisperx whisperx

# 复制项目文件
COPY pyproject.toml uv.lock README.md ./
COPY main.py ./
COPY config.py ./

# 安装 Python 依赖（使用 uv）
RUN uv sync --frozen --no-dev

# 从模型下载阶段复制预下载的模型
COPY --from=model-downloader /root/.cache /home/whisperx/.cache

# 创建必要的目录
RUN mkdir -p /app/temp /app/logs

# 设置权限
RUN chown -R whisperx:whisperx /app /home/whisperx

# 切换到非root用户
USER whisperx

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["uv", "run", "python", "main.py"] 
