FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

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
    # Python 3.12 和相关工具
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    python3-pip \
    # 清理工具
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建 Python 3.12 的符号链接
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# 安装 uv（工具安装，很少变化）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 创建缓存目录（在安装依赖之前创建）
RUN mkdir -p /root/.cache
VOLUME /root/.cache

# 复制项目依赖文件（只复制依赖文件，不复制源码）
COPY pyproject.toml uv.lock ./

# 安装 Python 依赖（使用 uv，指定 Python 3.12）
RUN uv sync --frozen --no-dev --python python3.12

# 复制项目源码（源码变化最频繁，放在最后）
COPY main.py ./
COPY src/ ./src/

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uv", "run", "python", "main.py"] 
