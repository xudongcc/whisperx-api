# =====================================
# 生产环境构建
# =====================================
FROM python:3.11-slim

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
    # Git（用于运行时下载模型）
    git \
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
