FROM ubuntu:22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖（这些依赖很少变化，放在前面利用缓存）
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
    # Python 3.10 和相关工具（Ubuntu 22.04 默认版本）
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # 清理工具
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 安装 uv（工具安装，很少变化）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 创建缓存目录（在安装依赖之前创建）
RUN mkdir -p /root/.cache
VOLUME /root/.cache

# 复制项目依赖文件
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖
RUN uv sync --no-dev

# 复制项目源码
COPY main.py ./
COPY src/ ./src/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "python", "main.py"] 
